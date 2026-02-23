"""
Panel-Corrected Standard Errors (PCSE).

PCSE (Beck & Katz 1995) are designed for panel data with cross-sectional
dependence. They estimate the full cross-sectional covariance matrix of
the errors and use FGLS to obtain efficient standard errors.

PCSE requires T > N (more time periods than entities).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)
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


@dataclass
class PCSEResult:
    """
    Result of PCSE estimation.

    Attributes
    ----------
    cov_matrix : np.ndarray
        PCSE covariance matrix (k x k)
    std_errors : np.ndarray
        PCSE standard errors (k,)
    sigma_matrix : np.ndarray
        Estimated cross-sectional error covariance matrix (N x N)
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters
    n_entities : int
        Number of entities
    n_periods : int
        Number of time periods
    """

    cov_matrix: np.ndarray
    std_errors: np.ndarray
    sigma_matrix: np.ndarray
    n_obs: int
    n_params: int
    n_entities: int
    n_periods: int


class PanelCorrectedStandardErrors:
    """
    Panel-Corrected Standard Errors (PCSE).

    Beck & Katz (1995) estimator for panel data with contemporaneous
    cross-sectional correlation. Estimates the full N×N contemporaneous
    covariance matrix and uses FGLS.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    entity_ids : np.ndarray
        Entity identifiers (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)

    Attributes
    ----------
    X : np.ndarray
        Design matrix
    resid : np.ndarray
        Residuals
    entity_ids : np.ndarray
        Entity identifiers
    time_ids : np.ndarray
        Time identifiers
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters
    n_entities : int
        Number of entities
    n_periods : int
        Number of time periods

    Examples
    --------
    >>> # Panel with T > N
    >>> pcse = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
    >>> result = pcse.compute()
    >>> print(result.std_errors)

    Notes
    -----
    PCSE requires T > N. If T < N, the estimated Σ matrix will be singular.

    References
    ----------
    Beck, N., & Katz, J. N. (1995). What to do (and not to do) with
        time-series cross-section data. American Political Science Review,
        89(3), 634-647.

    Bailey, D., & Katz, J. N. (2011). Implementing panel corrected standard
        errors in R: The pcse package. Journal of Statistical Software,
        42(CS1), 1-11.
    """

    def __init__(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        args = [X, resid, entity_ids, time_ids]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_orig"),
            object.__getattribute__(
                self, "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_mutants"
            ),
            args,
            kwargs,
            self,
        )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_orig(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_1(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = None
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_2(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = None
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_3(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = None
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_4(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(None)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_5(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = None

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_6(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(None)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_7(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = None

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_8(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) == self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_9(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(None)
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_10(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) == self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_11(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(None)

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_12(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = None
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_13(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(None)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_14(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = None
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_15(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(None)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_16(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = None
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_17(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = None

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_18(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods < self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_19(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                None,
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_20(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                None,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_21(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=None,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_22(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                UserWarning,
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_23(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_24(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning, stacklevel=2,
            )

    def xǁPanelCorrectedStandardErrorsǁ__init____mutmut_25(
        self, X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Get unique entities and periods
        self.unique_entities = np.unique(self.entity_ids)
        self.unique_times = np.unique(self.time_ids)
        self.n_entities = len(self.unique_entities)
        self.n_periods = len(self.unique_times)

        # Check T > N requirement
        if self.n_periods <= self.n_entities:
            import warnings

            warnings.warn(
                f"PCSE requires T > N. Got T={self.n_periods}, N={self.n_entities}. "
                f"The estimated Σ matrix may be singular or poorly estimated. "
                f"Consider using cluster-robust or Driscoll-Kraay SEs instead.",
                UserWarning,
                stacklevel=3,
            )

    xǁPanelCorrectedStandardErrorsǁ__init____mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_1": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_1,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_2": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_2,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_3": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_3,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_4": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_4,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_5": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_5,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_6": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_6,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_7": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_7,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_8": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_8,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_9": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_9,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_10": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_10,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_11": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_11,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_12": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_12,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_13": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_13,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_14": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_14,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_15": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_15,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_16": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_16,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_17": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_17,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_18": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_18,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_19": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_19,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_20": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_20,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_21": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_21,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_22": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_22,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_23": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_23,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_24": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_24,
        "xǁPanelCorrectedStandardErrorsǁ__init____mutmut_25": xǁPanelCorrectedStandardErrorsǁ__init____mutmut_25,
    }
    xǁPanelCorrectedStandardErrorsǁ__init____mutmut_orig.__name__ = (
        "xǁPanelCorrectedStandardErrorsǁ__init__"
    )

    def _reshape_panel(self) -> np.ndarray:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(
                self, "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_orig"
            ),
            object.__getattribute__(
                self, "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_mutants"
            ),
            args,
            kwargs,
            self,
        )

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_orig(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        time_map = {t: j for j, t in enumerate(self.unique_times)}

        # Initialize with NaN for unbalanced panels
        resid_matrix = np.full((self.n_entities, self.n_periods), np.nan)

        # Fill in observed values
        for i in range(self.n_obs):
            entity_idx = entity_map[self.entity_ids[i]]
            time_idx = time_map[self.time_ids[i]]
            resid_matrix[entity_idx, time_idx] = self.resid[i]

        return resid_matrix

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_1(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        entity_map = None
        time_map = {t: j for j, t in enumerate(self.unique_times)}

        # Initialize with NaN for unbalanced panels
        resid_matrix = np.full((self.n_entities, self.n_periods), np.nan)

        # Fill in observed values
        for i in range(self.n_obs):
            entity_idx = entity_map[self.entity_ids[i]]
            time_idx = time_map[self.time_ids[i]]
            resid_matrix[entity_idx, time_idx] = self.resid[i]

        return resid_matrix

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_2(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        entity_map = {e: i for i, e in enumerate(None)}
        time_map = {t: j for j, t in enumerate(self.unique_times)}

        # Initialize with NaN for unbalanced panels
        resid_matrix = np.full((self.n_entities, self.n_periods), np.nan)

        # Fill in observed values
        for i in range(self.n_obs):
            entity_idx = entity_map[self.entity_ids[i]]
            time_idx = time_map[self.time_ids[i]]
            resid_matrix[entity_idx, time_idx] = self.resid[i]

        return resid_matrix

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_3(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        time_map = None

        # Initialize with NaN for unbalanced panels
        resid_matrix = np.full((self.n_entities, self.n_periods), np.nan)

        # Fill in observed values
        for i in range(self.n_obs):
            entity_idx = entity_map[self.entity_ids[i]]
            time_idx = time_map[self.time_ids[i]]
            resid_matrix[entity_idx, time_idx] = self.resid[i]

        return resid_matrix

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_4(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        time_map = {t: j for j, t in enumerate(None)}

        # Initialize with NaN for unbalanced panels
        resid_matrix = np.full((self.n_entities, self.n_periods), np.nan)

        # Fill in observed values
        for i in range(self.n_obs):
            entity_idx = entity_map[self.entity_ids[i]]
            time_idx = time_map[self.time_ids[i]]
            resid_matrix[entity_idx, time_idx] = self.resid[i]

        return resid_matrix

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_5(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        time_map = {t: j for j, t in enumerate(self.unique_times)}

        # Initialize with NaN for unbalanced panels
        resid_matrix = None

        # Fill in observed values
        for i in range(self.n_obs):
            entity_idx = entity_map[self.entity_ids[i]]
            time_idx = time_map[self.time_ids[i]]
            resid_matrix[entity_idx, time_idx] = self.resid[i]

        return resid_matrix

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_6(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        time_map = {t: j for j, t in enumerate(self.unique_times)}

        # Initialize with NaN for unbalanced panels
        resid_matrix = np.full(None, np.nan)

        # Fill in observed values
        for i in range(self.n_obs):
            entity_idx = entity_map[self.entity_ids[i]]
            time_idx = time_map[self.time_ids[i]]
            resid_matrix[entity_idx, time_idx] = self.resid[i]

        return resid_matrix

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_7(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        time_map = {t: j for j, t in enumerate(self.unique_times)}

        # Initialize with NaN for unbalanced panels
        resid_matrix = np.full((self.n_entities, self.n_periods), None)

        # Fill in observed values
        for i in range(self.n_obs):
            entity_idx = entity_map[self.entity_ids[i]]
            time_idx = time_map[self.time_ids[i]]
            resid_matrix[entity_idx, time_idx] = self.resid[i]

        return resid_matrix

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_8(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        time_map = {t: j for j, t in enumerate(self.unique_times)}

        # Initialize with NaN for unbalanced panels
        resid_matrix = np.full(np.nan)

        # Fill in observed values
        for i in range(self.n_obs):
            entity_idx = entity_map[self.entity_ids[i]]
            time_idx = time_map[self.time_ids[i]]
            resid_matrix[entity_idx, time_idx] = self.resid[i]

        return resid_matrix

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_9(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        time_map = {t: j for j, t in enumerate(self.unique_times)}

        # Initialize with NaN for unbalanced panels
        resid_matrix = np.full(
            (self.n_entities, self.n_periods),
        )

        # Fill in observed values
        for i in range(self.n_obs):
            entity_idx = entity_map[self.entity_ids[i]]
            time_idx = time_map[self.time_ids[i]]
            resid_matrix[entity_idx, time_idx] = self.resid[i]

        return resid_matrix

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_10(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        time_map = {t: j for j, t in enumerate(self.unique_times)}

        # Initialize with NaN for unbalanced panels
        resid_matrix = np.full((self.n_entities, self.n_periods), np.nan)

        # Fill in observed values
        for i in range(None):
            entity_idx = entity_map[self.entity_ids[i]]
            time_idx = time_map[self.time_ids[i]]
            resid_matrix[entity_idx, time_idx] = self.resid[i]

        return resid_matrix

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_11(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        {e: i for i, e in enumerate(self.unique_entities)}
        time_map = {t: j for j, t in enumerate(self.unique_times)}

        # Initialize with NaN for unbalanced panels
        resid_matrix = np.full((self.n_entities, self.n_periods), np.nan)

        # Fill in observed values
        for i in range(self.n_obs):
            entity_idx = None
            time_idx = time_map[self.time_ids[i]]
            resid_matrix[entity_idx, time_idx] = self.resid[i]

        return resid_matrix

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_12(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        {t: j for j, t in enumerate(self.unique_times)}

        # Initialize with NaN for unbalanced panels
        resid_matrix = np.full((self.n_entities, self.n_periods), np.nan)

        # Fill in observed values
        for i in range(self.n_obs):
            entity_idx = entity_map[self.entity_ids[i]]
            time_idx = None
            resid_matrix[entity_idx, time_idx] = self.resid[i]

        return resid_matrix

    def xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_13(self) -> np.ndarray:
        """
        Reshape residuals to (N x T) matrix.

        Returns
        -------
        resid_matrix : np.ndarray
            Residuals reshaped to (N x T)
        """
        # Create mapping from entity to row index
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        time_map = {t: j for j, t in enumerate(self.unique_times)}

        # Initialize with NaN for unbalanced panels
        resid_matrix = np.full((self.n_entities, self.n_periods), np.nan)

        # Fill in observed values
        for i in range(self.n_obs):
            entity_idx = entity_map[self.entity_ids[i]]
            time_idx = time_map[self.time_ids[i]]
            resid_matrix[entity_idx, time_idx] = None

        return resid_matrix

    xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_1": xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_1,
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_2": xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_2,
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_3": xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_3,
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_4": xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_4,
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_5": xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_5,
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_6": xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_6,
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_7": xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_7,
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_8": xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_8,
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_9": xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_9,
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_10": xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_10,
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_11": xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_11,
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_12": xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_12,
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_13": xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_13,
    }
    xǁPanelCorrectedStandardErrorsǁ_reshape_panel__mutmut_orig.__name__ = (
        "xǁPanelCorrectedStandardErrorsǁ_reshape_panel"
    )

    def _estimate_sigma(self) -> np.ndarray:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(
                self, "xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_orig"
            ),
            object.__getattribute__(
                self, "xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_mutants"
            ),
            args,
            kwargs,
            self,
        )

    def xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_orig(self) -> np.ndarray:
        """
        Estimate contemporaneous covariance matrix Σ (N x N).

        Σ̂_ij = (1/T) Σ_t ε_it ε_jt

        Returns
        -------
        sigma : np.ndarray
            Estimated contemporaneous covariance matrix (N x N)
        """
        resid_matrix = self._reshape_panel()  # (N x T)

        # For balanced panels: Σ = (1/T) E E'
        # where E is the (N x T) residual matrix
        sigma = (resid_matrix @ resid_matrix.T) / self.n_periods

        # For unbalanced panels, need pairwise estimation
        # For now, we use simple approach with available data
        # More sophisticated: pairwise covariance with available pairs

        return sigma

    def xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_1(self) -> np.ndarray:
        """
        Estimate contemporaneous covariance matrix Σ (N x N).

        Σ̂_ij = (1/T) Σ_t ε_it ε_jt

        Returns
        -------
        sigma : np.ndarray
            Estimated contemporaneous covariance matrix (N x N)
        """
        resid_matrix = None  # (N x T)

        # For balanced panels: Σ = (1/T) E E'
        # where E is the (N x T) residual matrix
        sigma = (resid_matrix @ resid_matrix.T) / self.n_periods

        # For unbalanced panels, need pairwise estimation
        # For now, we use simple approach with available data
        # More sophisticated: pairwise covariance with available pairs

        return sigma

    def xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_2(self) -> np.ndarray:
        """
        Estimate contemporaneous covariance matrix Σ (N x N).

        Σ̂_ij = (1/T) Σ_t ε_it ε_jt

        Returns
        -------
        sigma : np.ndarray
            Estimated contemporaneous covariance matrix (N x N)
        """
        self._reshape_panel()  # (N x T)

        # For balanced panels: Σ = (1/T) E E'
        # where E is the (N x T) residual matrix
        sigma = None

        # For unbalanced panels, need pairwise estimation
        # For now, we use simple approach with available data
        # More sophisticated: pairwise covariance with available pairs

        return sigma

    def xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_3(self) -> np.ndarray:
        """
        Estimate contemporaneous covariance matrix Σ (N x N).

        Σ̂_ij = (1/T) Σ_t ε_it ε_jt

        Returns
        -------
        sigma : np.ndarray
            Estimated contemporaneous covariance matrix (N x N)
        """
        resid_matrix = self._reshape_panel()  # (N x T)

        # For balanced panels: Σ = (1/T) E E'
        # where E is the (N x T) residual matrix
        sigma = (resid_matrix @ resid_matrix.T) * self.n_periods

        # For unbalanced panels, need pairwise estimation
        # For now, we use simple approach with available data
        # More sophisticated: pairwise covariance with available pairs

        return sigma

    xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_1": xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_1,
        "xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_2": xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_2,
        "xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_3": xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_3,
    }
    xǁPanelCorrectedStandardErrorsǁ_estimate_sigma__mutmut_orig.__name__ = (
        "xǁPanelCorrectedStandardErrorsǁ_estimate_sigma"
    )

    def compute(self) -> PCSEResult:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_orig"),
            object.__getattribute__(self, "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_orig(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_1(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = None  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_2(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = None
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_3(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(None)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_4(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                None,
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_5(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                None,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_6(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=None,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_7(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_8(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_9(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning, stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_10(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "XXΣ matrix is singular. Using pseudoinverse. Results may be unreliable.XX",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_11(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "σ matrix is singular. using pseudoinverse. results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_12(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ MATRIX IS SINGULAR. USING PSEUDOINVERSE. RESULTS MAY BE UNRELIABLE.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_13(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=3,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_14(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = None

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_15(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(None)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_16(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = None
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_17(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(None)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_18(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = None

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_19(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array(None)

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_20(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = None

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_21(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = None

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_22(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros(None)

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_23(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(None):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_24(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(None):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_25(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = None
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_26(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = None

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_27(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_indices[i]
                entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = None

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_28(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX = weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_29(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX -= weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_30(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight / np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_31(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(None, self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_32(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], None)

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_33(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_34(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(
                    self.X[i],
                )

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_35(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = None
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_36(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(None)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_37(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(None, UserWarning, stacklevel=2)
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_38(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn("X'ΩX matrix is singular. Using pseudoinverse.", None, stacklevel=2)
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_39(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=None
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_40(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(UserWarning, stacklevel=2)
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_41(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn("X'ΩX matrix is singular. Using pseudoinverse.", stacklevel=2)
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_42(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.",
                UserWarning, stacklevel=2,
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_43(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "XXX'ΩX matrix is singular. Using pseudoinverse.XX", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_44(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "x'ωx matrix is singular. using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_45(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX MATRIX IS SINGULAR. USING PSEUDOINVERSE.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_46(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=3
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_47(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = None

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_48(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(None)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_49(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = None

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_50(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(None)

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_51(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(None))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_52(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=None,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_53(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=None,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_54(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=None,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_55(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=None,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_56(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=None,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_57(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=None,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_58(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=None,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_59(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_60(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_61(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_62(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_params=self.n_params,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_63(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_entities=self.n_entities,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_64(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁPanelCorrectedStandardErrorsǁcompute__mutmut_65(self) -> PCSEResult:
        """
        Compute PCSE covariance matrix.

        Returns
        -------
        result : PCSEResult
            PCSE covariance and standard errors

        Notes
        -----
        The PCSE estimator uses FGLS with estimated Σ:

        V_PCSE = (X' (Σ̂^{-1} ⊗ I_T) X)^{-1}

        where Σ̂ is the estimated N×N contemporaneous covariance matrix.

        For balanced panels:
        - Reshape residuals to (N x T) matrix E
        - Estimate Σ̂ = (1/T) E E'
        - Compute Ω = Σ̂ ⊗ I_T
        - V = (X' Ω^{-1} X)^{-1}
        """
        # Estimate Σ
        sigma = self._estimate_sigma()  # (N x N)

        # For large N, inverting Σ can be problematic
        # We use Moore-Penrose pseudoinverse for robustness
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Σ matrix is singular. Using pseudoinverse. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            sigma_inv = np.linalg.pinv(sigma)

        # Create Ω = Σ^{-1} ⊗ I_T
        # For efficiency, we don't explicitly form the full Ω matrix
        # Instead, we compute X' Ω^{-1} X directly

        # Create entity mapping
        entity_map = {e: i for i, e in enumerate(self.unique_entities)}
        entity_indices = np.array([entity_map[e] for e in self.entity_ids])

        # Compute weighted X: X_tilde = sqrt(Σ^{-1}) ⊗ I_T applied to X
        # This is done row-by-row based on entity
        k = self.n_params

        # Method: For each observation, weight by corresponding Σ^{-1} element
        # V = (Σ X'_i X_j)^{-1} where sum is over all pairs weighted by Σ^{-1}

        # More direct approach: X' Ω^{-1} X where Ω^{-1} = Σ^{-1} ⊗ I_T
        XtOmegaX = np.zeros((k, k))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                entity_i = entity_indices[i]
                entity_j = entity_indices[j]

                # Weight by Σ^{-1}[entity_i, entity_j]
                weight = sigma_inv[entity_i, entity_j]

                # Add contribution
                XtOmegaX += weight * np.outer(self.X[i], self.X[j])

        # Invert to get covariance
        try:
            cov_matrix = np.linalg.inv(XtOmegaX)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "X'ΩX matrix is singular. Using pseudoinverse.", UserWarning, stacklevel=2
            )
            cov_matrix = np.linalg.pinv(XtOmegaX)

        std_errors = np.sqrt(np.diag(cov_matrix))

        return PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_entities=self.n_entities,
        )

    xǁPanelCorrectedStandardErrorsǁcompute__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_1": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_1,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_2": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_2,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_3": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_3,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_4": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_4,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_5": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_5,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_6": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_6,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_7": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_7,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_8": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_8,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_9": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_9,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_10": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_10,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_11": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_11,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_12": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_12,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_13": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_13,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_14": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_14,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_15": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_15,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_16": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_16,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_17": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_17,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_18": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_18,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_19": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_19,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_20": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_20,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_21": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_21,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_22": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_22,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_23": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_23,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_24": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_24,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_25": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_25,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_26": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_26,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_27": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_27,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_28": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_28,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_29": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_29,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_30": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_30,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_31": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_31,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_32": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_32,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_33": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_33,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_34": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_34,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_35": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_35,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_36": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_36,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_37": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_37,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_38": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_38,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_39": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_39,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_40": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_40,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_41": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_41,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_42": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_42,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_43": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_43,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_44": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_44,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_45": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_45,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_46": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_46,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_47": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_47,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_48": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_48,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_49": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_49,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_50": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_50,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_51": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_51,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_52": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_52,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_53": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_53,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_54": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_54,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_55": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_55,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_56": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_56,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_57": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_57,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_58": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_58,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_59": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_59,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_60": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_60,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_61": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_61,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_62": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_62,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_63": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_63,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_64": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_64,
        "xǁPanelCorrectedStandardErrorsǁcompute__mutmut_65": xǁPanelCorrectedStandardErrorsǁcompute__mutmut_65,
    }
    xǁPanelCorrectedStandardErrorsǁcompute__mutmut_orig.__name__ = (
        "xǁPanelCorrectedStandardErrorsǁcompute"
    )

    def diagnostic_summary(self) -> str:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(
                self, "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_orig"
            ),
            object.__getattribute__(
                self, "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_mutants"
            ),
            args,
            kwargs,
            self,
        )

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_orig(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_1(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = None
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_2(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append(None)
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_3(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("XXPanel-Corrected Standard Errors DiagnosticsXX")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_4(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("panel-corrected standard errors diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_5(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("PANEL-CORRECTED STANDARD ERRORS DIAGNOSTICS")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_6(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append(None)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_7(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" / 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_8(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("XX=XX" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_9(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 51)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_10(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(None)
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_11(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(None)
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_12(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(None)
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_13(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(None)
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_14(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs * self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_15(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append(None)

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_16(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("XXXX")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_17(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods < self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_18(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append(None)
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_19(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("XX⚠ CRITICAL: T ≤ NXX")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_20(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ critical: t ≤ n")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_21(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append(None)
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_22(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("XX  PCSE requires T > NXX")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_23(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  pcse requires t > n")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_24(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE REQUIRES T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_25(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(None)
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_26(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append(None)
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_27(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("XX  Σ matrix will be poorly estimated or singularXX")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_28(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_29(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ MATRIX WILL BE POORLY ESTIMATED OR SINGULAR")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_30(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append(None)
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_31(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("XX  Consider cluster-robust or Driscoll-Kraay SEsXX")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_32(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  consider cluster-robust or driscoll-kraay ses")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_33(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  CONSIDER CLUSTER-ROBUST OR DRISCOLL-KRAAY SES")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_34(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods <= 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_35(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 / self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_36(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 3 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_37(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append(None)
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_38(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("XX⚠ WARNING: T < 2NXX")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_39(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ warning: t < 2n")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_40(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(None)
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_41(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append(None)
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_42(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("XX  PCSE may be unreliable with T/N < 2XX")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_43(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  pcse may be unreliable with t/n < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_44(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE MAY BE UNRELIABLE WITH T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_45(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(None)
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_46(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods * self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_47(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append(None)

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_48(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("XX  Sufficient for PCSE estimationXX")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_49(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  sufficient for pcse estimation")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_50(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  SUFFICIENT FOR PCSE ESTIMATION")

        return "\n".join(lines)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_51(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "\n".join(None)

    def xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_52(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Panel-Corrected Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of entities (N): {self.n_entities}")
        lines.append(f"Number of time periods (T): {self.n_periods}")
        lines.append(f"Average obs per entity: {self.n_obs / self.n_entities:.1f}")
        lines.append("")

        # Check requirements
        if self.n_periods <= self.n_entities:
            lines.append("⚠ CRITICAL: T ≤ N")
            lines.append("  PCSE requires T > N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  Σ matrix will be poorly estimated or singular")
            lines.append("  Consider cluster-robust or Driscoll-Kraay SEs")
        elif self.n_periods < 2 * self.n_entities:
            lines.append("⚠ WARNING: T < 2N")
            lines.append(f"  T={self.n_periods}, N={self.n_entities}")
            lines.append("  PCSE may be unreliable with T/N < 2")
        else:
            lines.append(f"✓ T/N ratio: {self.n_periods / self.n_entities:.2f}")
            lines.append("  Sufficient for PCSE estimation")

        return "XX\nXX".join(lines)

    xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_1": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_1,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_2": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_2,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_3": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_3,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_4": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_4,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_5": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_5,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_6": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_6,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_7": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_7,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_8": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_8,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_9": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_9,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_10": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_10,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_11": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_11,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_12": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_12,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_13": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_13,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_14": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_14,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_15": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_15,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_16": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_16,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_17": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_17,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_18": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_18,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_19": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_19,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_20": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_20,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_21": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_21,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_22": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_22,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_23": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_23,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_24": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_24,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_25": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_25,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_26": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_26,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_27": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_27,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_28": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_28,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_29": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_29,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_30": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_30,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_31": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_31,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_32": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_32,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_33": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_33,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_34": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_34,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_35": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_35,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_36": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_36,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_37": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_37,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_38": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_38,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_39": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_39,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_40": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_40,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_41": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_41,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_42": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_42,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_43": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_43,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_44": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_44,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_45": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_45,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_46": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_46,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_47": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_47,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_48": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_48,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_49": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_49,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_50": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_50,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_51": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_51,
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_52": xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_52,
    }
    xǁPanelCorrectedStandardErrorsǁdiagnostic_summary__mutmut_orig.__name__ = (
        "xǁPanelCorrectedStandardErrorsǁdiagnostic_summary"
    )


def pcse(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
) -> PCSEResult:
    args = [X, resid, entity_ids, time_ids]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(x_pcse__mutmut_orig, x_pcse__mutmut_mutants, args, kwargs, None)


def x_pcse__mutmut_orig(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
) -> PCSEResult:
    """
    Convenience function for Panel-Corrected Standard Errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    entity_ids : np.ndarray
        Entity identifiers (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)

    Returns
    -------
    result : PCSEResult
        PCSE covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import pcse
    >>> result = pcse(X, resid, entity_ids, time_ids)
    >>> print(result.std_errors)
    """
    pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
    return pcse_est.compute()


def x_pcse__mutmut_1(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
) -> PCSEResult:
    """
    Convenience function for Panel-Corrected Standard Errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    entity_ids : np.ndarray
        Entity identifiers (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)

    Returns
    -------
    result : PCSEResult
        PCSE covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import pcse
    >>> result = pcse(X, resid, entity_ids, time_ids)
    >>> print(result.std_errors)
    """
    pcse_est = None
    return pcse_est.compute()


def x_pcse__mutmut_2(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
) -> PCSEResult:
    """
    Convenience function for Panel-Corrected Standard Errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    entity_ids : np.ndarray
        Entity identifiers (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)

    Returns
    -------
    result : PCSEResult
        PCSE covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import pcse
    >>> result = pcse(X, resid, entity_ids, time_ids)
    >>> print(result.std_errors)
    """
    pcse_est = PanelCorrectedStandardErrors(None, resid, entity_ids, time_ids)
    return pcse_est.compute()


def x_pcse__mutmut_3(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
) -> PCSEResult:
    """
    Convenience function for Panel-Corrected Standard Errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    entity_ids : np.ndarray
        Entity identifiers (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)

    Returns
    -------
    result : PCSEResult
        PCSE covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import pcse
    >>> result = pcse(X, resid, entity_ids, time_ids)
    >>> print(result.std_errors)
    """
    pcse_est = PanelCorrectedStandardErrors(X, None, entity_ids, time_ids)
    return pcse_est.compute()


def x_pcse__mutmut_4(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
) -> PCSEResult:
    """
    Convenience function for Panel-Corrected Standard Errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    entity_ids : np.ndarray
        Entity identifiers (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)

    Returns
    -------
    result : PCSEResult
        PCSE covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import pcse
    >>> result = pcse(X, resid, entity_ids, time_ids)
    >>> print(result.std_errors)
    """
    pcse_est = PanelCorrectedStandardErrors(X, resid, None, time_ids)
    return pcse_est.compute()


def x_pcse__mutmut_5(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
) -> PCSEResult:
    """
    Convenience function for Panel-Corrected Standard Errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    entity_ids : np.ndarray
        Entity identifiers (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)

    Returns
    -------
    result : PCSEResult
        PCSE covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import pcse
    >>> result = pcse(X, resid, entity_ids, time_ids)
    >>> print(result.std_errors)
    """
    pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, None)
    return pcse_est.compute()


def x_pcse__mutmut_6(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
) -> PCSEResult:
    """
    Convenience function for Panel-Corrected Standard Errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    entity_ids : np.ndarray
        Entity identifiers (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)

    Returns
    -------
    result : PCSEResult
        PCSE covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import pcse
    >>> result = pcse(X, resid, entity_ids, time_ids)
    >>> print(result.std_errors)
    """
    pcse_est = PanelCorrectedStandardErrors(resid, entity_ids, time_ids)
    return pcse_est.compute()


def x_pcse__mutmut_7(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
) -> PCSEResult:
    """
    Convenience function for Panel-Corrected Standard Errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    entity_ids : np.ndarray
        Entity identifiers (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)

    Returns
    -------
    result : PCSEResult
        PCSE covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import pcse
    >>> result = pcse(X, resid, entity_ids, time_ids)
    >>> print(result.std_errors)
    """
    pcse_est = PanelCorrectedStandardErrors(X, entity_ids, time_ids)
    return pcse_est.compute()


def x_pcse__mutmut_8(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
) -> PCSEResult:
    """
    Convenience function for Panel-Corrected Standard Errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    entity_ids : np.ndarray
        Entity identifiers (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)

    Returns
    -------
    result : PCSEResult
        PCSE covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import pcse
    >>> result = pcse(X, resid, entity_ids, time_ids)
    >>> print(result.std_errors)
    """
    pcse_est = PanelCorrectedStandardErrors(X, resid, time_ids)
    return pcse_est.compute()


def x_pcse__mutmut_9(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, time_ids: np.ndarray
) -> PCSEResult:
    """
    Convenience function for Panel-Corrected Standard Errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    entity_ids : np.ndarray
        Entity identifiers (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)

    Returns
    -------
    result : PCSEResult
        PCSE covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import pcse
    >>> result = pcse(X, resid, entity_ids, time_ids)
    >>> print(result.std_errors)
    """
    pcse_est = PanelCorrectedStandardErrors(
        X,
        resid,
        entity_ids,
    )
    return pcse_est.compute()


x_pcse__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_pcse__mutmut_1": x_pcse__mutmut_1,
    "x_pcse__mutmut_2": x_pcse__mutmut_2,
    "x_pcse__mutmut_3": x_pcse__mutmut_3,
    "x_pcse__mutmut_4": x_pcse__mutmut_4,
    "x_pcse__mutmut_5": x_pcse__mutmut_5,
    "x_pcse__mutmut_6": x_pcse__mutmut_6,
    "x_pcse__mutmut_7": x_pcse__mutmut_7,
    "x_pcse__mutmut_8": x_pcse__mutmut_8,
    "x_pcse__mutmut_9": x_pcse__mutmut_9,
}
x_pcse__mutmut_orig.__name__ = "x_pcse"
