"""
Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors.

Newey-West (1987) standard errors are robust to both heteroskedasticity and
autocorrelation. Useful for time-series and panel data with serial correlation.
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
class NeweyWestResult:
    """
    Result of Newey-West HAC covariance estimation.

    Attributes
    ----------
    cov_matrix : np.ndarray
        Newey-West covariance matrix (k x k)
    std_errors : np.ndarray
        Newey-West standard errors (k,)
    max_lags : int
        Maximum number of lags used
    kernel : str
        Kernel function used
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters
    prewhitening : bool
        Whether prewhitening was applied
    """

    cov_matrix: np.ndarray
    std_errors: np.ndarray
    max_lags: int
    kernel: str
    n_obs: int
    n_params: int
    prewhitening: bool = False


class NeweyWestStandardErrors:
    """
    Newey-West (1987) HAC standard errors.

    Robust to heteroskedasticity and autocorrelation. Particularly useful
    for time-series data and panel data with serial correlation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags. If None, uses floor(4(T/100)^(2/9))
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function for weighting lags
    prewhitening : bool, default=False
        Apply AR(1) prewhitening to reduce finite-sample bias

    Attributes
    ----------
    X : np.ndarray
        Design matrix
    resid : np.ndarray
        Residuals
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters

    Examples
    --------
    >>> # Time-series with autocorrelation
    >>> nw = NeweyWestStandardErrors(X, resid, max_lags=4)
    >>> result = nw.compute()
    >>> print(result.std_errors)

    >>> # Auto-select lags
    >>> nw = NeweyWestStandardErrors(X, resid)
    >>> result = nw.compute()

    References
    ----------
    Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
        heteroskedasticity and autocorrelation consistent covariance matrix.
        Econometrica, 55(3), 703-708.

    Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent
        covariance matrix estimation. Econometrica, 59(3), 817-858.
    """

    def __init__(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        args = [X, resid, max_lags, kernel, prewhitening]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁNeweyWestStandardErrorsǁ__init____mutmut_orig"),
            object.__getattribute__(self, "xǁNeweyWestStandardErrorsǁ__init____mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_orig(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_1(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "XXbartlettXX",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_2(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "BARTLETT",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_3(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = True,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_4(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = None
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_5(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = None
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_6(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = None
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_7(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = None

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_8(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = None

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_9(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is not None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_10(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = None
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_11(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(None)
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_12(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(None))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_13(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 / (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_14(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(5 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_15(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) * (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_16(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs * 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_17(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 101) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_18(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 * 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_19(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (3 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_20(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 10)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_21(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = None

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_22(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags > self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_23(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = None

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_24(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs + 1

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_25(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 2

        # Cache
        self._bread: np.ndarray | None = None

    def xǁNeweyWestStandardErrorsǁ__init____mutmut_26(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: int | None = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: np.ndarray | None = ""

    xǁNeweyWestStandardErrorsǁ__init____mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_1": xǁNeweyWestStandardErrorsǁ__init____mutmut_1,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_2": xǁNeweyWestStandardErrorsǁ__init____mutmut_2,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_3": xǁNeweyWestStandardErrorsǁ__init____mutmut_3,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_4": xǁNeweyWestStandardErrorsǁ__init____mutmut_4,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_5": xǁNeweyWestStandardErrorsǁ__init____mutmut_5,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_6": xǁNeweyWestStandardErrorsǁ__init____mutmut_6,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_7": xǁNeweyWestStandardErrorsǁ__init____mutmut_7,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_8": xǁNeweyWestStandardErrorsǁ__init____mutmut_8,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_9": xǁNeweyWestStandardErrorsǁ__init____mutmut_9,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_10": xǁNeweyWestStandardErrorsǁ__init____mutmut_10,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_11": xǁNeweyWestStandardErrorsǁ__init____mutmut_11,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_12": xǁNeweyWestStandardErrorsǁ__init____mutmut_12,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_13": xǁNeweyWestStandardErrorsǁ__init____mutmut_13,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_14": xǁNeweyWestStandardErrorsǁ__init____mutmut_14,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_15": xǁNeweyWestStandardErrorsǁ__init____mutmut_15,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_16": xǁNeweyWestStandardErrorsǁ__init____mutmut_16,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_17": xǁNeweyWestStandardErrorsǁ__init____mutmut_17,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_18": xǁNeweyWestStandardErrorsǁ__init____mutmut_18,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_19": xǁNeweyWestStandardErrorsǁ__init____mutmut_19,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_20": xǁNeweyWestStandardErrorsǁ__init____mutmut_20,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_21": xǁNeweyWestStandardErrorsǁ__init____mutmut_21,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_22": xǁNeweyWestStandardErrorsǁ__init____mutmut_22,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_23": xǁNeweyWestStandardErrorsǁ__init____mutmut_23,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_24": xǁNeweyWestStandardErrorsǁ__init____mutmut_24,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_25": xǁNeweyWestStandardErrorsǁ__init____mutmut_25,
        "xǁNeweyWestStandardErrorsǁ__init____mutmut_26": xǁNeweyWestStandardErrorsǁ__init____mutmut_26,
    }
    xǁNeweyWestStandardErrorsǁ__init____mutmut_orig.__name__ = "xǁNeweyWestStandardErrorsǁ__init__"

    @property
    def bread(self) -> np.ndarray:
        """Compute and cache bread matrix."""
        if self._bread is None:
            self._bread = compute_bread(self.X)
        return self._bread

    def _kernel_weight(self, lag: int) -> float:
        args = [lag]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_orig"),
            object.__getattribute__(
                self, "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_mutants"
            ),
            args,
            kwargs,
            self,
        )

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_orig(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_1(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_2(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_3(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_4(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_5(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_6(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_7(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_8(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_9(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_10(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_11(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_12(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_13(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_14(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_15(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_16(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_17(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_18(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_19(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_20(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_21(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_22(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_23(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_24(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_25(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_26(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_27(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_28(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_29(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_30(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_31(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_32(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_33(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_34(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_35(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_36(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_37(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_38(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_39(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_40(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_41(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_42(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_43(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_44(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_45(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_46(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_47(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_48(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_49(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_50(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_51(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_52(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_53(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_54(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_55(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_56(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_57(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_58(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_59(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_60(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_61(self, lag: int) -> float:
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

    def xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_62(self, lag: int) -> float:
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

    xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_1": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_1,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_2": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_2,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_3": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_3,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_4": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_4,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_5": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_5,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_6": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_6,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_7": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_7,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_8": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_8,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_9": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_9,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_10": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_10,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_11": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_11,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_12": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_12,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_13": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_13,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_14": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_14,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_15": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_15,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_16": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_16,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_17": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_17,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_18": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_18,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_19": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_19,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_20": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_20,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_21": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_21,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_22": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_22,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_23": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_23,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_24": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_24,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_25": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_25,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_26": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_26,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_27": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_27,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_28": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_28,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_29": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_29,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_30": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_30,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_31": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_31,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_32": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_32,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_33": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_33,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_34": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_34,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_35": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_35,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_36": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_36,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_37": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_37,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_38": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_38,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_39": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_39,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_40": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_40,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_41": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_41,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_42": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_42,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_43": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_43,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_44": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_44,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_45": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_45,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_46": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_46,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_47": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_47,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_48": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_48,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_49": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_49,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_50": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_50,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_51": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_51,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_52": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_52,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_53": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_53,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_54": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_54,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_55": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_55,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_56": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_56,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_57": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_57,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_58": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_58,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_59": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_59,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_60": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_60,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_61": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_61,
        "xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_62": xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_62,
    }
    xǁNeweyWestStandardErrorsǁ_kernel_weight__mutmut_orig.__name__ = (
        "xǁNeweyWestStandardErrorsǁ_kernel_weight"
    )

    def _compute_gamma(self, lag: int) -> np.ndarray:
        args = [lag]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_orig"),
            object.__getattribute__(
                self, "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_mutants"
            ),
            args,
            kwargs,
            self,
        )

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_orig(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] * self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_1(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = None

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] * self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_2(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag != 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] * self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_3(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 1:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] * self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_4(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = None
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] * self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_5(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X / self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] * self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_6(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            self.X * self.resid[:, np.newaxis]
            gamma = None
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] * self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_7(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) * n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] * self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_8(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = None
            X_resid_t_lag = self.X[:-lag] * self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_9(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] / self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] * self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_10(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = None
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_11(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] / self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_12(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:+lag] * self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_13(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] * self.resid[:+lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_14(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            self.X[lag:] * self.resid[lag:, np.newaxis]
            self.X[:-lag] * self.resid[:-lag, np.newaxis]
            gamma = None

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_15(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] * self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) * n

        return np.asarray(gamma)

    def xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_16(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] * self.resid[:-lag, np.newaxis]
            (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(None)

    xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_1": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_1,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_2": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_2,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_3": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_3,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_4": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_4,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_5": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_5,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_6": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_6,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_7": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_7,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_8": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_8,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_9": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_9,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_10": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_10,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_11": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_11,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_12": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_12,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_13": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_13,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_14": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_14,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_15": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_15,
        "xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_16": xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_16,
    }
    xǁNeweyWestStandardErrorsǁ_compute_gamma__mutmut_orig.__name__ = (
        "xǁNeweyWestStandardErrorsǁ_compute_gamma"
    )

    def compute(self) -> NeweyWestResult:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁNeweyWestStandardErrorsǁcompute__mutmut_orig"),
            object.__getattribute__(self, "xǁNeweyWestStandardErrorsǁcompute__mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_orig(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_1(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = None

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_2(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(None)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_3(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(1)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_4(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(None, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_5(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, None):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_6(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_7(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
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

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_8(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(2, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_9(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags - 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_10(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 2):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_11(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = None
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_12(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(None)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_13(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight >= 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_14(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 1:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_15(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = None
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_16(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(None)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_17(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S = weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_18(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S -= weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_19(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight / (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_20(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l - gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_21(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S = self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_22(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S /= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_23(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = None
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_24(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(None, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_25(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, None)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_26(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_27(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(
            self.bread,
        )
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_28(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = None

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_29(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(None)

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_30(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(None))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_31(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=None,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_32(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=None,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_33(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=None,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_34(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=None,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_35(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=None,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_36(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=None,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_37(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=None,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_38(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_39(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_40(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_41(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_42(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_43(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            prewhitening=self.prewhitening,
        )

    def xǁNeweyWestStandardErrorsǁcompute__mutmut_44(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    xǁNeweyWestStandardErrorsǁcompute__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_1": xǁNeweyWestStandardErrorsǁcompute__mutmut_1,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_2": xǁNeweyWestStandardErrorsǁcompute__mutmut_2,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_3": xǁNeweyWestStandardErrorsǁcompute__mutmut_3,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_4": xǁNeweyWestStandardErrorsǁcompute__mutmut_4,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_5": xǁNeweyWestStandardErrorsǁcompute__mutmut_5,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_6": xǁNeweyWestStandardErrorsǁcompute__mutmut_6,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_7": xǁNeweyWestStandardErrorsǁcompute__mutmut_7,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_8": xǁNeweyWestStandardErrorsǁcompute__mutmut_8,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_9": xǁNeweyWestStandardErrorsǁcompute__mutmut_9,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_10": xǁNeweyWestStandardErrorsǁcompute__mutmut_10,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_11": xǁNeweyWestStandardErrorsǁcompute__mutmut_11,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_12": xǁNeweyWestStandardErrorsǁcompute__mutmut_12,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_13": xǁNeweyWestStandardErrorsǁcompute__mutmut_13,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_14": xǁNeweyWestStandardErrorsǁcompute__mutmut_14,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_15": xǁNeweyWestStandardErrorsǁcompute__mutmut_15,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_16": xǁNeweyWestStandardErrorsǁcompute__mutmut_16,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_17": xǁNeweyWestStandardErrorsǁcompute__mutmut_17,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_18": xǁNeweyWestStandardErrorsǁcompute__mutmut_18,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_19": xǁNeweyWestStandardErrorsǁcompute__mutmut_19,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_20": xǁNeweyWestStandardErrorsǁcompute__mutmut_20,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_21": xǁNeweyWestStandardErrorsǁcompute__mutmut_21,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_22": xǁNeweyWestStandardErrorsǁcompute__mutmut_22,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_23": xǁNeweyWestStandardErrorsǁcompute__mutmut_23,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_24": xǁNeweyWestStandardErrorsǁcompute__mutmut_24,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_25": xǁNeweyWestStandardErrorsǁcompute__mutmut_25,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_26": xǁNeweyWestStandardErrorsǁcompute__mutmut_26,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_27": xǁNeweyWestStandardErrorsǁcompute__mutmut_27,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_28": xǁNeweyWestStandardErrorsǁcompute__mutmut_28,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_29": xǁNeweyWestStandardErrorsǁcompute__mutmut_29,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_30": xǁNeweyWestStandardErrorsǁcompute__mutmut_30,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_31": xǁNeweyWestStandardErrorsǁcompute__mutmut_31,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_32": xǁNeweyWestStandardErrorsǁcompute__mutmut_32,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_33": xǁNeweyWestStandardErrorsǁcompute__mutmut_33,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_34": xǁNeweyWestStandardErrorsǁcompute__mutmut_34,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_35": xǁNeweyWestStandardErrorsǁcompute__mutmut_35,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_36": xǁNeweyWestStandardErrorsǁcompute__mutmut_36,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_37": xǁNeweyWestStandardErrorsǁcompute__mutmut_37,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_38": xǁNeweyWestStandardErrorsǁcompute__mutmut_38,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_39": xǁNeweyWestStandardErrorsǁcompute__mutmut_39,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_40": xǁNeweyWestStandardErrorsǁcompute__mutmut_40,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_41": xǁNeweyWestStandardErrorsǁcompute__mutmut_41,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_42": xǁNeweyWestStandardErrorsǁcompute__mutmut_42,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_43": xǁNeweyWestStandardErrorsǁcompute__mutmut_43,
        "xǁNeweyWestStandardErrorsǁcompute__mutmut_44": xǁNeweyWestStandardErrorsǁcompute__mutmut_44,
    }
    xǁNeweyWestStandardErrorsǁcompute__mutmut_orig.__name__ = "xǁNeweyWestStandardErrorsǁcompute"

    def diagnostic_summary(self) -> str:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(
                self, "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_orig"
            ),
            object.__getattribute__(
                self, "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_mutants"
            ),
            args,
            kwargs,
            self,
        )

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_orig(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_1(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = None
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_2(self) -> str:
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
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_3(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("XXNewey-West HAC Standard Errors DiagnosticsXX")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_4(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("newey-west hac standard errors diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_5(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("NEWEY-WEST HAC STANDARD ERRORS DIAGNOSTICS")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_6(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append(None)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_7(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" / 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_8(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("XX=XX" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_9(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 51)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_10(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(None)
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_11(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(None)
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_12(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(None)
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_13(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(None)
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_14(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(None)
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_15(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append(None)

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_16(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("XXXX")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_17(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs <= 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_18(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 51:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_19(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append(None)
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_20(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("XX⚠ WARNING: Small sample size (<50)XX")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_21(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ warning: small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_22(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: SMALL SAMPLE SIZE (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_23(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append(None)
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_24(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("XX  Newey-West SEs may not perform well with few observationsXX")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_25(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  newey-west ses may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_26(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  NEWEY-WEST SES MAY NOT PERFORM WELL WITH FEW OBSERVATIONS")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_27(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags >= self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_28(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs * 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_29(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 4:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_30(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append(None)
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_31(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("XX⚠ WARNING: Large max_lags relative to sample sizeXX")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_32(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ warning: large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_33(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: LARGE MAX_LAGS RELATIVE TO SAMPLE SIZE")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_34(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(None)

        return "\n".join(lines)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_35(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(None)

    def xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_36(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "XX\nXX".join(lines)

    xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_1": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_1,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_2": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_2,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_3": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_3,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_4": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_4,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_5": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_5,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_6": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_6,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_7": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_7,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_8": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_8,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_9": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_9,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_10": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_10,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_11": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_11,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_12": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_12,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_13": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_13,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_14": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_14,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_15": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_15,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_16": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_16,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_17": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_17,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_18": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_18,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_19": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_19,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_20": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_20,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_21": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_21,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_22": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_22,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_23": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_23,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_24": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_24,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_25": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_25,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_26": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_26,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_27": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_27,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_28": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_28,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_29": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_29,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_30": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_30,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_31": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_31,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_32": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_32,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_33": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_33,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_34": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_34,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_35": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_35,
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_36": xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_36,
    }
    xǁNeweyWestStandardErrorsǁdiagnostic_summary__mutmut_orig.__name__ = (
        "xǁNeweyWestStandardErrorsǁdiagnostic_summary"
    )


def newey_west(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    args = [X, resid, max_lags, kernel, prewhitening]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_newey_west__mutmut_orig, x_newey_west__mutmut_mutants, args, kwargs, None
    )


def x_newey_west__mutmut_orig(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(X, resid, max_lags, kernel, prewhitening)
    return nw.compute()


def x_newey_west__mutmut_1(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "XXbartlettXX",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(X, resid, max_lags, kernel, prewhitening)
    return nw.compute()


def x_newey_west__mutmut_2(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "BARTLETT",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(X, resid, max_lags, kernel, prewhitening)
    return nw.compute()


def x_newey_west__mutmut_3(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = True,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(X, resid, max_lags, kernel, prewhitening)
    return nw.compute()


def x_newey_west__mutmut_4(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = None
    return nw.compute()


def x_newey_west__mutmut_5(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(None, resid, max_lags, kernel, prewhitening)
    return nw.compute()


def x_newey_west__mutmut_6(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(X, None, max_lags, kernel, prewhitening)
    return nw.compute()


def x_newey_west__mutmut_7(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(X, resid, None, kernel, prewhitening)
    return nw.compute()


def x_newey_west__mutmut_8(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(X, resid, max_lags, None, prewhitening)
    return nw.compute()


def x_newey_west__mutmut_9(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(X, resid, max_lags, kernel, None)
    return nw.compute()


def x_newey_west__mutmut_10(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(resid, max_lags, kernel, prewhitening)
    return nw.compute()


def x_newey_west__mutmut_11(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(X, max_lags, kernel, prewhitening)
    return nw.compute()


def x_newey_west__mutmut_12(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(X, resid, kernel, prewhitening)
    return nw.compute()


def x_newey_west__mutmut_13(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(X, resid, max_lags, prewhitening)
    return nw.compute()


def x_newey_west__mutmut_14(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(
        X,
        resid,
        max_lags,
        kernel,
    )
    return nw.compute()


x_newey_west__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_newey_west__mutmut_1": x_newey_west__mutmut_1,
    "x_newey_west__mutmut_2": x_newey_west__mutmut_2,
    "x_newey_west__mutmut_3": x_newey_west__mutmut_3,
    "x_newey_west__mutmut_4": x_newey_west__mutmut_4,
    "x_newey_west__mutmut_5": x_newey_west__mutmut_5,
    "x_newey_west__mutmut_6": x_newey_west__mutmut_6,
    "x_newey_west__mutmut_7": x_newey_west__mutmut_7,
    "x_newey_west__mutmut_8": x_newey_west__mutmut_8,
    "x_newey_west__mutmut_9": x_newey_west__mutmut_9,
    "x_newey_west__mutmut_10": x_newey_west__mutmut_10,
    "x_newey_west__mutmut_11": x_newey_west__mutmut_11,
    "x_newey_west__mutmut_12": x_newey_west__mutmut_12,
    "x_newey_west__mutmut_13": x_newey_west__mutmut_13,
    "x_newey_west__mutmut_14": x_newey_west__mutmut_14,
}
x_newey_west__mutmut_orig.__name__ = "x_newey_west"
