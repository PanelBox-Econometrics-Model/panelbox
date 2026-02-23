"""
Driscoll-Kraay standard errors for panel data.

Driscoll-Kraay (1998) standard errors are robust to general forms of
spatial and temporal dependence when the number of time periods is large.
They are particularly useful for macro panel data with potential cross-
sectional correlation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .utils import compute_bread, sandwich_covariance

logger = logging.getLogger(__name__)

KernelType = Literal["bartlett", "parzen", "quadratic_spectral"]
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
class DriscollKraayResult:
    """
    Result of Driscoll-Kraay covariance estimation.

    Attributes
    ----------
    cov_matrix : np.ndarray
        Driscoll-Kraay covariance matrix (k x k)
    std_errors : np.ndarray
        Driscoll-Kraay standard errors (k,)
    max_lags : int
        Maximum number of lags used
    kernel : str
        Kernel function used
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters
    n_periods : int
        Number of time periods
    bandwidth : Optional[float]
        Bandwidth parameter (for some kernels)
    """

    cov_matrix: np.ndarray
    std_errors: np.ndarray
    max_lags: int
    kernel: str
    n_obs: int
    n_params: int
    n_periods: int
    bandwidth: float | None = None


class DriscollKraayStandardErrors:
    """
    Driscoll-Kraay (1998) standard errors for panel data.

    Robust to general forms of spatial and temporal dependence.
    Particularly useful for macro panels with cross-sectional correlation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags. If None, uses floor(4(T/100)^(2/9))
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function for weighting lags

    Attributes
    ----------
    X : np.ndarray
        Design matrix
    resid : np.ndarray
        Residuals
    time_ids : np.ndarray
        Time identifiers
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters
    n_periods : int
        Number of time periods

    Examples
    --------
    >>> # Panel data with T=20 periods
    >>> dk = DriscollKraayStandardErrors(X, resid, time_ids)
    >>> result = dk.compute()
    >>> print(result.std_errors)

    >>> # Custom lags
    >>> dk = DriscollKraayStandardErrors(X, resid, time_ids, max_lags=5)
    >>> result = dk.compute()

    References
    ----------
    Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix
        estimation with spatially dependent panel data. Review of Economics
        and Statistics, 80(4), 549-560.

    Hoechle, D. (2007). Robust standard errors for panel regressions with
        cross-sectional dependence. The Stata Journal, 7(3), 281-312.
    """

    def __init__(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        args = [X, resid, time_ids, max_lags, kernel]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁDriscollKraayStandardErrorsǁ__init____mutmut_orig"),
            object.__getattribute__(self, "xǁDriscollKraayStandardErrorsǁ__init____mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_orig(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_1(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "XXbartlettXX",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_2(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "BARTLETT",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_3(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = None
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_4(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = None
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_5(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = None
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_6(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(None)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_7(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = None

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_8(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = None

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_9(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) == self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_10(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(None)

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_11(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = None
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_12(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(None)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_13(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        np.unique(self.time_ids)
        self.n_periods = None

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_14(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is not None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_15(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = None
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_16(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(None)
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_17(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(None))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_18(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 / (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_19(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(5 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_20(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) * (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_21(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods * 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_22(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 101) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_23(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 * 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_24(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (3 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_25(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 10)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_26(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = None

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_27(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags > self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_28(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = None

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_29(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods + 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_30(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 2

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_31(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = ""
        self._time_sorted: dict | None = None

    def xǁDriscollKraayStandardErrorsǁ__init____mutmut_32(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread: np.ndarray | None = None
        self._time_sorted: dict | None = ""

    xǁDriscollKraayStandardErrorsǁ__init____mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_1": xǁDriscollKraayStandardErrorsǁ__init____mutmut_1,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_2": xǁDriscollKraayStandardErrorsǁ__init____mutmut_2,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_3": xǁDriscollKraayStandardErrorsǁ__init____mutmut_3,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_4": xǁDriscollKraayStandardErrorsǁ__init____mutmut_4,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_5": xǁDriscollKraayStandardErrorsǁ__init____mutmut_5,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_6": xǁDriscollKraayStandardErrorsǁ__init____mutmut_6,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_7": xǁDriscollKraayStandardErrorsǁ__init____mutmut_7,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_8": xǁDriscollKraayStandardErrorsǁ__init____mutmut_8,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_9": xǁDriscollKraayStandardErrorsǁ__init____mutmut_9,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_10": xǁDriscollKraayStandardErrorsǁ__init____mutmut_10,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_11": xǁDriscollKraayStandardErrorsǁ__init____mutmut_11,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_12": xǁDriscollKraayStandardErrorsǁ__init____mutmut_12,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_13": xǁDriscollKraayStandardErrorsǁ__init____mutmut_13,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_14": xǁDriscollKraayStandardErrorsǁ__init____mutmut_14,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_15": xǁDriscollKraayStandardErrorsǁ__init____mutmut_15,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_16": xǁDriscollKraayStandardErrorsǁ__init____mutmut_16,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_17": xǁDriscollKraayStandardErrorsǁ__init____mutmut_17,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_18": xǁDriscollKraayStandardErrorsǁ__init____mutmut_18,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_19": xǁDriscollKraayStandardErrorsǁ__init____mutmut_19,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_20": xǁDriscollKraayStandardErrorsǁ__init____mutmut_20,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_21": xǁDriscollKraayStandardErrorsǁ__init____mutmut_21,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_22": xǁDriscollKraayStandardErrorsǁ__init____mutmut_22,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_23": xǁDriscollKraayStandardErrorsǁ__init____mutmut_23,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_24": xǁDriscollKraayStandardErrorsǁ__init____mutmut_24,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_25": xǁDriscollKraayStandardErrorsǁ__init____mutmut_25,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_26": xǁDriscollKraayStandardErrorsǁ__init____mutmut_26,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_27": xǁDriscollKraayStandardErrorsǁ__init____mutmut_27,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_28": xǁDriscollKraayStandardErrorsǁ__init____mutmut_28,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_29": xǁDriscollKraayStandardErrorsǁ__init____mutmut_29,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_30": xǁDriscollKraayStandardErrorsǁ__init____mutmut_30,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_31": xǁDriscollKraayStandardErrorsǁ__init____mutmut_31,
        "xǁDriscollKraayStandardErrorsǁ__init____mutmut_32": xǁDriscollKraayStandardErrorsǁ__init____mutmut_32,
    }
    xǁDriscollKraayStandardErrorsǁ__init____mutmut_orig.__name__ = (
        "xǁDriscollKraayStandardErrorsǁ__init__"
    )

    @property
    def bread(self) -> np.ndarray:
        """Compute and cache bread matrix."""
        if self._bread is None:
            self._bread = compute_bread(self.X)
        return self._bread

    def _sort_by_time(self):
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(
                self, "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_orig"
            ),
            object.__getattribute__(
                self, "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_mutants"
            ),
            args,
            kwargs,
            self,
        )

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_orig(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_1(self):
        """Sort data by time periods."""
        if self._time_sorted is not None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_2(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = None

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_3(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(None)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_4(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = None

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_5(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(None)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_6(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = None
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_7(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array(None)
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_8(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            np.array([time_map[t] for t in self.time_ids])
            sort_idx = None

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_9(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(None)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_10(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            np.argsort(time_indices)

            self._time_sorted = None

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_11(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "XXXXX": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_12(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "x": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_13(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "XXresidXX": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_14(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "RESID": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_15(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "XXtime_idsXX": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_16(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "TIME_IDS": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_17(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "XXsort_idxXX": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_18(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "SORT_IDX": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_19(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "XXunique_timesXX": unique_times,
            }

        return self._time_sorted

    def xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_20(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "UNIQUE_TIMES": unique_times,
            }

        return self._time_sorted

    xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_1": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_1,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_2": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_2,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_3": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_3,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_4": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_4,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_5": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_5,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_6": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_6,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_7": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_7,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_8": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_8,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_9": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_9,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_10": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_10,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_11": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_11,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_12": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_12,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_13": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_13,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_14": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_14,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_15": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_15,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_16": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_16,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_17": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_17,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_18": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_18,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_19": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_19,
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_20": xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_20,
    }
    xǁDriscollKraayStandardErrorsǁ_sort_by_time__mutmut_orig.__name__ = (
        "xǁDriscollKraayStandardErrorsǁ_sort_by_time"
    )

    def _kernel_weight(self, lag: int) -> float:
        args = [lag]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(
                self, "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_orig"
            ),
            object.__getattribute__(
                self, "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_mutants"
            ),
            args,
            kwargs,
            self,
        )

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_orig(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_1(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag >= self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_2(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 1.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_3(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel != "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_4(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "XXbartlettXX":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_5(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "BARTLETT":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_6(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 + lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_7(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 2.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_8(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag * (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_9(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags - 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_10(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 2)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_11(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel != "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_12(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "XXparzenXX":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_13(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "PARZEN":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_14(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = None
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_15(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag * (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_16(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags - 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_17(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 2)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_18(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z < 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_19(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 1.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_20(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 - 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_21(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 + 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_22(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 2 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_23(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 / z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_24(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 7 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_25(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z * 2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_26(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**3 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_27(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 / z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_28(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 7 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_29(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z * 3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_30(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**4
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_31(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 / (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_32(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 3 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_33(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) * 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_34(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 + z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_35(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (2 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_36(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 4

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_37(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel != "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_38(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "XXquadratic_spectralXX":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_39(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "QUADRATIC_SPECTRAL":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_40(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag != 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_41(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 1:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_42(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 2.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_43(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = None
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_44(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) * 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_45(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag * (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_46(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi / lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_47(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 / np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_48(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 7 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_49(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags - 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_50(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 2) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_51(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 6
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_52(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(None)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_53(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 / (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_54(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 * z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_55(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(4 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_56(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z * 2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_57(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**3 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_58(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z + np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_59(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) * z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_60(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(None) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_61(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(None)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_62(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(None)

    xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_1": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_1,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_2": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_2,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_3": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_3,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_4": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_4,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_5": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_5,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_6": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_6,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_7": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_7,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_8": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_8,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_9": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_9,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_10": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_10,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_11": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_11,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_12": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_12,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_13": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_13,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_14": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_14,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_15": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_15,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_16": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_16,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_17": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_17,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_18": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_18,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_19": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_19,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_20": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_20,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_21": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_21,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_22": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_22,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_23": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_23,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_24": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_24,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_25": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_25,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_26": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_26,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_27": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_27,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_28": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_28,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_29": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_29,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_30": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_30,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_31": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_31,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_32": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_32,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_33": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_33,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_34": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_34,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_35": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_35,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_36": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_36,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_37": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_37,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_38": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_38,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_39": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_39,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_40": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_40,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_41": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_41,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_42": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_42,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_43": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_43,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_44": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_44,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_45": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_45,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_46": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_46,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_47": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_47,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_48": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_48,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_49": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_49,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_50": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_50,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_51": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_51,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_52": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_52,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_53": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_53,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_54": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_54,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_55": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_55,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_56": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_56,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_57": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_57,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_58": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_58,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_59": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_59,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_60": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_60,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_61": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_61,
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_62": xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_62,
    }
    xǁDriscollKraayStandardErrorsǁ_kernel_weight__mutmut_orig.__name__ = (
        "xǁDriscollKraayStandardErrorsǁ_kernel_weight"
    )

    def _compute_gamma(self, lag: int) -> np.ndarray:
        args = [lag]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(
                self, "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_orig"
            ),
            object.__getattribute__(
                self, "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_mutants"
            ),
            args,
            kwargs,
            self,
        )

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_orig(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_1(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = None
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_2(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = None
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_3(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["XXunique_timesXX"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_4(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["UNIQUE_TIMES"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_5(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = None

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_6(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]

        gamma = None

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_7(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]

        gamma = np.zeros(None)

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_8(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(None, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_9(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, None):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_10(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_11(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(
            lag,
        ):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_12(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = None
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_13(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = None

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_14(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx + lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_15(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = None
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_16(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["XXtime_idsXX"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_17(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["TIME_IDS"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_18(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] != t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_19(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = None
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_20(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["XXXXX"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_21(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["x"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_22(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = None

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_23(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["XXresidXX"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_24(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["RESID"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_25(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = None
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_26(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["XXtime_idsXX"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_27(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["TIME_IDS"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_28(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] != t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_29(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = None
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_30(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["XXXXX"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_31(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["x"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_32(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = None

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_33(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["XXresidXX"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_34(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["RESID"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_35(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(None):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_36(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(None):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_37(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma = np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_38(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma -= np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_39(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for _i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(None, X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_40(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for _j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], None)

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_41(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for _i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_42(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for _j in range(len(X_t_lag)):
                    gamma += np.outer(
                        X_t[i] * resid_t[i],
                    )

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_43(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] / resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_44(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] / resid_t_lag[j])

        return gamma

    xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_1": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_1,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_2": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_2,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_3": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_3,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_4": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_4,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_5": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_5,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_6": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_6,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_7": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_7,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_8": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_8,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_9": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_9,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_10": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_10,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_11": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_11,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_12": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_12,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_13": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_13,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_14": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_14,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_15": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_15,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_16": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_16,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_17": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_17,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_18": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_18,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_19": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_19,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_20": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_20,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_21": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_21,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_22": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_22,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_23": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_23,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_24": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_24,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_25": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_25,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_26": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_26,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_27": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_27,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_28": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_28,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_29": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_29,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_30": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_30,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_31": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_31,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_32": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_32,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_33": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_33,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_34": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_34,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_35": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_35,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_36": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_36,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_37": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_37,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_38": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_38,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_39": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_39,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_40": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_40,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_41": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_41,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_42": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_42,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_43": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_43,
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_44": xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_44,
    }
    xǁDriscollKraayStandardErrorsǁ_compute_gamma__mutmut_orig.__name__ = (
        "xǁDriscollKraayStandardErrorsǁ_compute_gamma"
    )

    def compute(self) -> DriscollKraayResult:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁDriscollKraayStandardErrorsǁcompute__mutmut_orig"),
            object.__getattribute__(self, "xǁDriscollKraayStandardErrorsǁcompute__mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_orig(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_1(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = None

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_2(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(None)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_3(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(1)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_4(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(None, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_5(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, None):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_6(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_7(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(
            1,
        ):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_8(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(2, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_9(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags - 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_10(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 2):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_11(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = None
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_12(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(None)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_13(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight >= 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_14(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 1:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_15(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = None
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_16(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(None)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_17(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S = weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_18(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S -= weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_19(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight / (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_20(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l - gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_21(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = None
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_22(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(None, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_23(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, None)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_24(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_25(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(
            self.bread,
        )
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_26(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = None

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_27(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(None)

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_28(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(None))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_29(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=None,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_30(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=None,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_31(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=None,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_32(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=None,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_33(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=None,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_34(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=None,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_35(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=None,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_36(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_37(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_38(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_39(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_40(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_41(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_periods=self.n_periods,
        )

    def xǁDriscollKraayStandardErrorsǁcompute__mutmut_42(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    xǁDriscollKraayStandardErrorsǁcompute__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_1": xǁDriscollKraayStandardErrorsǁcompute__mutmut_1,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_2": xǁDriscollKraayStandardErrorsǁcompute__mutmut_2,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_3": xǁDriscollKraayStandardErrorsǁcompute__mutmut_3,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_4": xǁDriscollKraayStandardErrorsǁcompute__mutmut_4,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_5": xǁDriscollKraayStandardErrorsǁcompute__mutmut_5,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_6": xǁDriscollKraayStandardErrorsǁcompute__mutmut_6,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_7": xǁDriscollKraayStandardErrorsǁcompute__mutmut_7,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_8": xǁDriscollKraayStandardErrorsǁcompute__mutmut_8,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_9": xǁDriscollKraayStandardErrorsǁcompute__mutmut_9,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_10": xǁDriscollKraayStandardErrorsǁcompute__mutmut_10,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_11": xǁDriscollKraayStandardErrorsǁcompute__mutmut_11,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_12": xǁDriscollKraayStandardErrorsǁcompute__mutmut_12,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_13": xǁDriscollKraayStandardErrorsǁcompute__mutmut_13,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_14": xǁDriscollKraayStandardErrorsǁcompute__mutmut_14,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_15": xǁDriscollKraayStandardErrorsǁcompute__mutmut_15,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_16": xǁDriscollKraayStandardErrorsǁcompute__mutmut_16,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_17": xǁDriscollKraayStandardErrorsǁcompute__mutmut_17,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_18": xǁDriscollKraayStandardErrorsǁcompute__mutmut_18,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_19": xǁDriscollKraayStandardErrorsǁcompute__mutmut_19,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_20": xǁDriscollKraayStandardErrorsǁcompute__mutmut_20,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_21": xǁDriscollKraayStandardErrorsǁcompute__mutmut_21,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_22": xǁDriscollKraayStandardErrorsǁcompute__mutmut_22,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_23": xǁDriscollKraayStandardErrorsǁcompute__mutmut_23,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_24": xǁDriscollKraayStandardErrorsǁcompute__mutmut_24,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_25": xǁDriscollKraayStandardErrorsǁcompute__mutmut_25,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_26": xǁDriscollKraayStandardErrorsǁcompute__mutmut_26,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_27": xǁDriscollKraayStandardErrorsǁcompute__mutmut_27,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_28": xǁDriscollKraayStandardErrorsǁcompute__mutmut_28,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_29": xǁDriscollKraayStandardErrorsǁcompute__mutmut_29,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_30": xǁDriscollKraayStandardErrorsǁcompute__mutmut_30,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_31": xǁDriscollKraayStandardErrorsǁcompute__mutmut_31,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_32": xǁDriscollKraayStandardErrorsǁcompute__mutmut_32,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_33": xǁDriscollKraayStandardErrorsǁcompute__mutmut_33,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_34": xǁDriscollKraayStandardErrorsǁcompute__mutmut_34,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_35": xǁDriscollKraayStandardErrorsǁcompute__mutmut_35,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_36": xǁDriscollKraayStandardErrorsǁcompute__mutmut_36,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_37": xǁDriscollKraayStandardErrorsǁcompute__mutmut_37,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_38": xǁDriscollKraayStandardErrorsǁcompute__mutmut_38,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_39": xǁDriscollKraayStandardErrorsǁcompute__mutmut_39,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_40": xǁDriscollKraayStandardErrorsǁcompute__mutmut_40,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_41": xǁDriscollKraayStandardErrorsǁcompute__mutmut_41,
        "xǁDriscollKraayStandardErrorsǁcompute__mutmut_42": xǁDriscollKraayStandardErrorsǁcompute__mutmut_42,
    }
    xǁDriscollKraayStandardErrorsǁcompute__mutmut_orig.__name__ = (
        "xǁDriscollKraayStandardErrorsǁcompute"
    )

    def diagnostic_summary(self) -> str:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(
                self, "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_orig"
            ),
            object.__getattribute__(
                self, "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_mutants"
            ),
            args,
            kwargs,
            self,
        )

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_orig(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_1(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = None
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_2(self) -> str:
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
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_3(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("XXDriscoll-Kraay Standard Errors DiagnosticsXX")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_4(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("driscoll-kraay standard errors diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_5(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("DRISCOLL-KRAAY STANDARD ERRORS DIAGNOSTICS")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_6(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append(None)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_7(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" / 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_8(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("XX=XX" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_9(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 51)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_10(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(None)
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_11(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(None)
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_12(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(None)
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_13(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs * self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_14(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(None)
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_15(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(None)
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_16(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(None)

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_17(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("XXXX")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_18(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods <= 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_19(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 21:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_20(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append(None)
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_21(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("XX⚠ WARNING: Few time periods (<20)XX")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_22(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ warning: few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_23(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: FEW TIME PERIODS (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_24(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append(None)
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_25(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("XX  Driscoll-Kraay SEs may not perform well with T < 20XX")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_26(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  driscoll-kraay ses may not perform well with t < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_27(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  DRISCOLL-KRAAY SES MAY NOT PERFORM WELL WITH T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_28(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append(None)
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_29(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("XX  Consider alternative methodsXX")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_30(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_31(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  CONSIDER ALTERNATIVE METHODS")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_32(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags >= self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_33(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods * 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_34(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 5:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_35(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append(None)
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_36(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("XX⚠ WARNING: Large max_lags relative to TXX")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_37(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ warning: large max_lags relative to t")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_38(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: LARGE MAX_LAGS RELATIVE TO T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_39(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(None)

        return "\n".join(lines)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_40(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(None)

    def xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_41(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "XX\nXX".join(lines)

    xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_1": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_1,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_2": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_2,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_3": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_3,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_4": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_4,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_5": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_5,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_6": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_6,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_7": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_7,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_8": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_8,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_9": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_9,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_10": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_10,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_11": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_11,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_12": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_12,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_13": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_13,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_14": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_14,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_15": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_15,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_16": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_16,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_17": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_17,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_18": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_18,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_19": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_19,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_20": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_20,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_21": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_21,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_22": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_22,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_23": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_23,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_24": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_24,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_25": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_25,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_26": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_26,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_27": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_27,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_28": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_28,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_29": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_29,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_30": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_30,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_31": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_31,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_32": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_32,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_33": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_33,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_34": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_34,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_35": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_35,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_36": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_36,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_37": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_37,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_38": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_38,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_39": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_39,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_40": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_40,
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_41": xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_41,
    }
    xǁDriscollKraayStandardErrorsǁdiagnostic_summary__mutmut_orig.__name__ = (
        "xǁDriscollKraayStandardErrorsǁdiagnostic_summary"
    )


def driscoll_kraay(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    args = [X, resid, time_ids, max_lags, kernel]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_driscoll_kraay__mutmut_orig, x_driscoll_kraay__mutmut_mutants, args, kwargs, None
    )


def x_driscoll_kraay__mutmut_orig(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(X, resid, time_ids, max_lags, kernel)
    return dk.compute()


def x_driscoll_kraay__mutmut_1(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "XXbartlettXX",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(X, resid, time_ids, max_lags, kernel)
    return dk.compute()


def x_driscoll_kraay__mutmut_2(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "BARTLETT",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(X, resid, time_ids, max_lags, kernel)
    return dk.compute()


def x_driscoll_kraay__mutmut_3(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = None
    return dk.compute()


def x_driscoll_kraay__mutmut_4(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(None, resid, time_ids, max_lags, kernel)
    return dk.compute()


def x_driscoll_kraay__mutmut_5(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(X, None, time_ids, max_lags, kernel)
    return dk.compute()


def x_driscoll_kraay__mutmut_6(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(X, resid, None, max_lags, kernel)
    return dk.compute()


def x_driscoll_kraay__mutmut_7(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(X, resid, time_ids, None, kernel)
    return dk.compute()


def x_driscoll_kraay__mutmut_8(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(X, resid, time_ids, max_lags, None)
    return dk.compute()


def x_driscoll_kraay__mutmut_9(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(resid, time_ids, max_lags, kernel)
    return dk.compute()


def x_driscoll_kraay__mutmut_10(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(X, time_ids, max_lags, kernel)
    return dk.compute()


def x_driscoll_kraay__mutmut_11(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(X, resid, max_lags, kernel)
    return dk.compute()


def x_driscoll_kraay__mutmut_12(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(X, resid, time_ids, kernel)
    return dk.compute()


def x_driscoll_kraay__mutmut_13(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(
        X,
        resid,
        time_ids,
        max_lags,
    )
    return dk.compute()


x_driscoll_kraay__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_driscoll_kraay__mutmut_1": x_driscoll_kraay__mutmut_1,
    "x_driscoll_kraay__mutmut_2": x_driscoll_kraay__mutmut_2,
    "x_driscoll_kraay__mutmut_3": x_driscoll_kraay__mutmut_3,
    "x_driscoll_kraay__mutmut_4": x_driscoll_kraay__mutmut_4,
    "x_driscoll_kraay__mutmut_5": x_driscoll_kraay__mutmut_5,
    "x_driscoll_kraay__mutmut_6": x_driscoll_kraay__mutmut_6,
    "x_driscoll_kraay__mutmut_7": x_driscoll_kraay__mutmut_7,
    "x_driscoll_kraay__mutmut_8": x_driscoll_kraay__mutmut_8,
    "x_driscoll_kraay__mutmut_9": x_driscoll_kraay__mutmut_9,
    "x_driscoll_kraay__mutmut_10": x_driscoll_kraay__mutmut_10,
    "x_driscoll_kraay__mutmut_11": x_driscoll_kraay__mutmut_11,
    "x_driscoll_kraay__mutmut_12": x_driscoll_kraay__mutmut_12,
    "x_driscoll_kraay__mutmut_13": x_driscoll_kraay__mutmut_13,
}
x_driscoll_kraay__mutmut_orig.__name__ = "x_driscoll_kraay"
