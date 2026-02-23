"""
Heteroskedasticity-robust standard errors (HC0, HC1, HC2, HC3).

This module implements White's heteroskedasticity-robust covariance
estimators and their finite-sample improvements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .utils import compute_bread, compute_leverage, compute_meat_hc, sandwich_covariance

logger = logging.getLogger(__name__)

HC_TYPES = Literal["HC0", "HC1", "HC2", "HC3"]
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
class RobustCovarianceResult:
    """
    Result of robust covariance estimation.

    Attributes
    ----------
    cov_matrix : np.ndarray
        Robust covariance matrix (k x k)
    std_errors : np.ndarray
        Robust standard errors (k,)
    method : str
        Method used ('HC0', 'HC1', 'HC2', 'HC3')
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters
    leverage : np.ndarray, optional
        Leverage values (for HC2, HC3)
    """

    cov_matrix: np.ndarray
    std_errors: np.ndarray
    method: str
    n_obs: int
    n_params: int
    leverage: np.ndarray | None = None


class RobustStandardErrors:
    """
    Heteroskedasticity-robust standard errors.

    Implements White (1980) and improved finite-sample variants.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)

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

    Methods
    -------
    hc0()
        White (1980) heteroskedasticity-robust SE
    hc1()
        Degrees of freedom corrected
    hc2()
        Leverage-adjusted
    hc3()
        MacKinnon-White (1985)

    Examples
    --------
    >>> from panelbox.standard_errors import RobustStandardErrors
    >>> robust = RobustStandardErrors(X, resid)
    >>> result_hc1 = robust.hc1()
    >>> print(result_hc1.std_errors)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.

    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """

    def __init__(self, X: np.ndarray, resid: np.ndarray):
        args = [X, resid]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁRobustStandardErrorsǁ__init____mutmut_orig"),
            object.__getattribute__(self, "xǁRobustStandardErrorsǁ__init____mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁRobustStandardErrorsǁ__init____mutmut_orig(self, X: np.ndarray, resid: np.ndarray):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape

        # Cache for efficiency
        self._leverage: np.ndarray | None = None
        self._bread: np.ndarray | None = None

    def xǁRobustStandardErrorsǁ__init____mutmut_1(self, X: np.ndarray, resid: np.ndarray):
        self.X = None
        self.resid = resid
        self.n_obs, self.n_params = X.shape

        # Cache for efficiency
        self._leverage: np.ndarray | None = None
        self._bread: np.ndarray | None = None

    def xǁRobustStandardErrorsǁ__init____mutmut_2(self, X: np.ndarray, resid: np.ndarray):
        self.X = X
        self.resid = None
        self.n_obs, self.n_params = X.shape

        # Cache for efficiency
        self._leverage: np.ndarray | None = None
        self._bread: np.ndarray | None = None

    def xǁRobustStandardErrorsǁ__init____mutmut_3(self, X: np.ndarray, resid: np.ndarray):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = None

        # Cache for efficiency
        self._leverage: np.ndarray | None = None
        self._bread: np.ndarray | None = None

    def xǁRobustStandardErrorsǁ__init____mutmut_4(self, X: np.ndarray, resid: np.ndarray):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape

        # Cache for efficiency
        self._leverage: np.ndarray | None = ""
        self._bread: np.ndarray | None = None

    def xǁRobustStandardErrorsǁ__init____mutmut_5(self, X: np.ndarray, resid: np.ndarray):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape

        # Cache for efficiency
        self._leverage: np.ndarray | None = None
        self._bread: np.ndarray | None = ""

    xǁRobustStandardErrorsǁ__init____mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁRobustStandardErrorsǁ__init____mutmut_1": xǁRobustStandardErrorsǁ__init____mutmut_1,
        "xǁRobustStandardErrorsǁ__init____mutmut_2": xǁRobustStandardErrorsǁ__init____mutmut_2,
        "xǁRobustStandardErrorsǁ__init____mutmut_3": xǁRobustStandardErrorsǁ__init____mutmut_3,
        "xǁRobustStandardErrorsǁ__init____mutmut_4": xǁRobustStandardErrorsǁ__init____mutmut_4,
        "xǁRobustStandardErrorsǁ__init____mutmut_5": xǁRobustStandardErrorsǁ__init____mutmut_5,
    }
    xǁRobustStandardErrorsǁ__init____mutmut_orig.__name__ = "xǁRobustStandardErrorsǁ__init__"

    @property
    def leverage(self) -> np.ndarray:
        """Compute and cache leverage values."""
        if self._leverage is None:
            self._leverage = compute_leverage(self.X)
        return self._leverage

    @property
    def bread(self) -> np.ndarray:
        """Compute and cache bread matrix."""
        if self._bread is None:
            self._bread = compute_bread(self.X)
        return self._bread

    def hc0(self) -> RobustCovarianceResult:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁRobustStandardErrorsǁhc0__mutmut_orig"),
            object.__getattribute__(self, "xǁRobustStandardErrorsǁhc0__mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_orig(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_1(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = None
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_2(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(None, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_3(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, None, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_4(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method=None)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_5(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_6(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_7(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(
            self.X,
            self.resid,
        )
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_8(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="XXHC0XX")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_9(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="hc0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_10(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = None
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_11(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(None, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_12(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, None)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_13(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_14(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(
            self.bread,
        )
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_15(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = None

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_16(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(None)

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_17(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(None))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_18(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=None,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_19(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=None,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_20(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method=None,
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_21(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=None,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_22(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=None,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_23(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_24(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_25(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_26(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_27(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_28(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="XXHC0XX",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc0__mutmut_29(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="hc0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    xǁRobustStandardErrorsǁhc0__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁRobustStandardErrorsǁhc0__mutmut_1": xǁRobustStandardErrorsǁhc0__mutmut_1,
        "xǁRobustStandardErrorsǁhc0__mutmut_2": xǁRobustStandardErrorsǁhc0__mutmut_2,
        "xǁRobustStandardErrorsǁhc0__mutmut_3": xǁRobustStandardErrorsǁhc0__mutmut_3,
        "xǁRobustStandardErrorsǁhc0__mutmut_4": xǁRobustStandardErrorsǁhc0__mutmut_4,
        "xǁRobustStandardErrorsǁhc0__mutmut_5": xǁRobustStandardErrorsǁhc0__mutmut_5,
        "xǁRobustStandardErrorsǁhc0__mutmut_6": xǁRobustStandardErrorsǁhc0__mutmut_6,
        "xǁRobustStandardErrorsǁhc0__mutmut_7": xǁRobustStandardErrorsǁhc0__mutmut_7,
        "xǁRobustStandardErrorsǁhc0__mutmut_8": xǁRobustStandardErrorsǁhc0__mutmut_8,
        "xǁRobustStandardErrorsǁhc0__mutmut_9": xǁRobustStandardErrorsǁhc0__mutmut_9,
        "xǁRobustStandardErrorsǁhc0__mutmut_10": xǁRobustStandardErrorsǁhc0__mutmut_10,
        "xǁRobustStandardErrorsǁhc0__mutmut_11": xǁRobustStandardErrorsǁhc0__mutmut_11,
        "xǁRobustStandardErrorsǁhc0__mutmut_12": xǁRobustStandardErrorsǁhc0__mutmut_12,
        "xǁRobustStandardErrorsǁhc0__mutmut_13": xǁRobustStandardErrorsǁhc0__mutmut_13,
        "xǁRobustStandardErrorsǁhc0__mutmut_14": xǁRobustStandardErrorsǁhc0__mutmut_14,
        "xǁRobustStandardErrorsǁhc0__mutmut_15": xǁRobustStandardErrorsǁhc0__mutmut_15,
        "xǁRobustStandardErrorsǁhc0__mutmut_16": xǁRobustStandardErrorsǁhc0__mutmut_16,
        "xǁRobustStandardErrorsǁhc0__mutmut_17": xǁRobustStandardErrorsǁhc0__mutmut_17,
        "xǁRobustStandardErrorsǁhc0__mutmut_18": xǁRobustStandardErrorsǁhc0__mutmut_18,
        "xǁRobustStandardErrorsǁhc0__mutmut_19": xǁRobustStandardErrorsǁhc0__mutmut_19,
        "xǁRobustStandardErrorsǁhc0__mutmut_20": xǁRobustStandardErrorsǁhc0__mutmut_20,
        "xǁRobustStandardErrorsǁhc0__mutmut_21": xǁRobustStandardErrorsǁhc0__mutmut_21,
        "xǁRobustStandardErrorsǁhc0__mutmut_22": xǁRobustStandardErrorsǁhc0__mutmut_22,
        "xǁRobustStandardErrorsǁhc0__mutmut_23": xǁRobustStandardErrorsǁhc0__mutmut_23,
        "xǁRobustStandardErrorsǁhc0__mutmut_24": xǁRobustStandardErrorsǁhc0__mutmut_24,
        "xǁRobustStandardErrorsǁhc0__mutmut_25": xǁRobustStandardErrorsǁhc0__mutmut_25,
        "xǁRobustStandardErrorsǁhc0__mutmut_26": xǁRobustStandardErrorsǁhc0__mutmut_26,
        "xǁRobustStandardErrorsǁhc0__mutmut_27": xǁRobustStandardErrorsǁhc0__mutmut_27,
        "xǁRobustStandardErrorsǁhc0__mutmut_28": xǁRobustStandardErrorsǁhc0__mutmut_28,
        "xǁRobustStandardErrorsǁhc0__mutmut_29": xǁRobustStandardErrorsǁhc0__mutmut_29,
    }
    xǁRobustStandardErrorsǁhc0__mutmut_orig.__name__ = "xǁRobustStandardErrorsǁhc0"

    def hc1(self) -> RobustCovarianceResult:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁRobustStandardErrorsǁhc1__mutmut_orig"),
            object.__getattribute__(self, "xǁRobustStandardErrorsǁhc1__mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_orig(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_1(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = None
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_2(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(None, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_3(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, None, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_4(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method=None)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_5(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_6(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_7(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(
            self.X,
            self.resid,
        )
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_8(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="XXHC1XX")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_9(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="hc1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_10(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = None
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_11(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(None, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_12(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, None)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_13(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_14(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(
            self.bread,
        )
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_15(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = None

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_16(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(None)

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_17(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(None))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_18(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=None,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_19(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=None,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_20(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method=None,
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_21(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=None,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_22(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=None,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_23(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_24(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_25(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_26(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_27(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_28(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="XXHC1XX",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc1__mutmut_29(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="hc1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    xǁRobustStandardErrorsǁhc1__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁRobustStandardErrorsǁhc1__mutmut_1": xǁRobustStandardErrorsǁhc1__mutmut_1,
        "xǁRobustStandardErrorsǁhc1__mutmut_2": xǁRobustStandardErrorsǁhc1__mutmut_2,
        "xǁRobustStandardErrorsǁhc1__mutmut_3": xǁRobustStandardErrorsǁhc1__mutmut_3,
        "xǁRobustStandardErrorsǁhc1__mutmut_4": xǁRobustStandardErrorsǁhc1__mutmut_4,
        "xǁRobustStandardErrorsǁhc1__mutmut_5": xǁRobustStandardErrorsǁhc1__mutmut_5,
        "xǁRobustStandardErrorsǁhc1__mutmut_6": xǁRobustStandardErrorsǁhc1__mutmut_6,
        "xǁRobustStandardErrorsǁhc1__mutmut_7": xǁRobustStandardErrorsǁhc1__mutmut_7,
        "xǁRobustStandardErrorsǁhc1__mutmut_8": xǁRobustStandardErrorsǁhc1__mutmut_8,
        "xǁRobustStandardErrorsǁhc1__mutmut_9": xǁRobustStandardErrorsǁhc1__mutmut_9,
        "xǁRobustStandardErrorsǁhc1__mutmut_10": xǁRobustStandardErrorsǁhc1__mutmut_10,
        "xǁRobustStandardErrorsǁhc1__mutmut_11": xǁRobustStandardErrorsǁhc1__mutmut_11,
        "xǁRobustStandardErrorsǁhc1__mutmut_12": xǁRobustStandardErrorsǁhc1__mutmut_12,
        "xǁRobustStandardErrorsǁhc1__mutmut_13": xǁRobustStandardErrorsǁhc1__mutmut_13,
        "xǁRobustStandardErrorsǁhc1__mutmut_14": xǁRobustStandardErrorsǁhc1__mutmut_14,
        "xǁRobustStandardErrorsǁhc1__mutmut_15": xǁRobustStandardErrorsǁhc1__mutmut_15,
        "xǁRobustStandardErrorsǁhc1__mutmut_16": xǁRobustStandardErrorsǁhc1__mutmut_16,
        "xǁRobustStandardErrorsǁhc1__mutmut_17": xǁRobustStandardErrorsǁhc1__mutmut_17,
        "xǁRobustStandardErrorsǁhc1__mutmut_18": xǁRobustStandardErrorsǁhc1__mutmut_18,
        "xǁRobustStandardErrorsǁhc1__mutmut_19": xǁRobustStandardErrorsǁhc1__mutmut_19,
        "xǁRobustStandardErrorsǁhc1__mutmut_20": xǁRobustStandardErrorsǁhc1__mutmut_20,
        "xǁRobustStandardErrorsǁhc1__mutmut_21": xǁRobustStandardErrorsǁhc1__mutmut_21,
        "xǁRobustStandardErrorsǁhc1__mutmut_22": xǁRobustStandardErrorsǁhc1__mutmut_22,
        "xǁRobustStandardErrorsǁhc1__mutmut_23": xǁRobustStandardErrorsǁhc1__mutmut_23,
        "xǁRobustStandardErrorsǁhc1__mutmut_24": xǁRobustStandardErrorsǁhc1__mutmut_24,
        "xǁRobustStandardErrorsǁhc1__mutmut_25": xǁRobustStandardErrorsǁhc1__mutmut_25,
        "xǁRobustStandardErrorsǁhc1__mutmut_26": xǁRobustStandardErrorsǁhc1__mutmut_26,
        "xǁRobustStandardErrorsǁhc1__mutmut_27": xǁRobustStandardErrorsǁhc1__mutmut_27,
        "xǁRobustStandardErrorsǁhc1__mutmut_28": xǁRobustStandardErrorsǁhc1__mutmut_28,
        "xǁRobustStandardErrorsǁhc1__mutmut_29": xǁRobustStandardErrorsǁhc1__mutmut_29,
    }
    xǁRobustStandardErrorsǁhc1__mutmut_orig.__name__ = "xǁRobustStandardErrorsǁhc1"

    def hc2(self) -> RobustCovarianceResult:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁRobustStandardErrorsǁhc2__mutmut_orig"),
            object.__getattribute__(self, "xǁRobustStandardErrorsǁhc2__mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_orig(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_1(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = None
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_2(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = None
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_3(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(None, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_4(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, None, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_5(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method=None, leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_6(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=None)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_7(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_8(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_9(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_10(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(
            self.X,
            self.resid,
            method="HC2",
        )
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_11(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="XXHC2XX", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_12(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="hc2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_13(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = None
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_14(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(None, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_15(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, None)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_16(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_17(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(
            self.bread,
        )
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_18(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = None

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_19(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(None)

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_20(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(None))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_21(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=None,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_22(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=None,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_23(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method=None,
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_24(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=None,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_25(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=None,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_26(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=None,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_27(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_28(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_29(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_30(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_31(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_32(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_33(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="XXHC2XX",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc2__mutmut_34(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="hc2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    xǁRobustStandardErrorsǁhc2__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁRobustStandardErrorsǁhc2__mutmut_1": xǁRobustStandardErrorsǁhc2__mutmut_1,
        "xǁRobustStandardErrorsǁhc2__mutmut_2": xǁRobustStandardErrorsǁhc2__mutmut_2,
        "xǁRobustStandardErrorsǁhc2__mutmut_3": xǁRobustStandardErrorsǁhc2__mutmut_3,
        "xǁRobustStandardErrorsǁhc2__mutmut_4": xǁRobustStandardErrorsǁhc2__mutmut_4,
        "xǁRobustStandardErrorsǁhc2__mutmut_5": xǁRobustStandardErrorsǁhc2__mutmut_5,
        "xǁRobustStandardErrorsǁhc2__mutmut_6": xǁRobustStandardErrorsǁhc2__mutmut_6,
        "xǁRobustStandardErrorsǁhc2__mutmut_7": xǁRobustStandardErrorsǁhc2__mutmut_7,
        "xǁRobustStandardErrorsǁhc2__mutmut_8": xǁRobustStandardErrorsǁhc2__mutmut_8,
        "xǁRobustStandardErrorsǁhc2__mutmut_9": xǁRobustStandardErrorsǁhc2__mutmut_9,
        "xǁRobustStandardErrorsǁhc2__mutmut_10": xǁRobustStandardErrorsǁhc2__mutmut_10,
        "xǁRobustStandardErrorsǁhc2__mutmut_11": xǁRobustStandardErrorsǁhc2__mutmut_11,
        "xǁRobustStandardErrorsǁhc2__mutmut_12": xǁRobustStandardErrorsǁhc2__mutmut_12,
        "xǁRobustStandardErrorsǁhc2__mutmut_13": xǁRobustStandardErrorsǁhc2__mutmut_13,
        "xǁRobustStandardErrorsǁhc2__mutmut_14": xǁRobustStandardErrorsǁhc2__mutmut_14,
        "xǁRobustStandardErrorsǁhc2__mutmut_15": xǁRobustStandardErrorsǁhc2__mutmut_15,
        "xǁRobustStandardErrorsǁhc2__mutmut_16": xǁRobustStandardErrorsǁhc2__mutmut_16,
        "xǁRobustStandardErrorsǁhc2__mutmut_17": xǁRobustStandardErrorsǁhc2__mutmut_17,
        "xǁRobustStandardErrorsǁhc2__mutmut_18": xǁRobustStandardErrorsǁhc2__mutmut_18,
        "xǁRobustStandardErrorsǁhc2__mutmut_19": xǁRobustStandardErrorsǁhc2__mutmut_19,
        "xǁRobustStandardErrorsǁhc2__mutmut_20": xǁRobustStandardErrorsǁhc2__mutmut_20,
        "xǁRobustStandardErrorsǁhc2__mutmut_21": xǁRobustStandardErrorsǁhc2__mutmut_21,
        "xǁRobustStandardErrorsǁhc2__mutmut_22": xǁRobustStandardErrorsǁhc2__mutmut_22,
        "xǁRobustStandardErrorsǁhc2__mutmut_23": xǁRobustStandardErrorsǁhc2__mutmut_23,
        "xǁRobustStandardErrorsǁhc2__mutmut_24": xǁRobustStandardErrorsǁhc2__mutmut_24,
        "xǁRobustStandardErrorsǁhc2__mutmut_25": xǁRobustStandardErrorsǁhc2__mutmut_25,
        "xǁRobustStandardErrorsǁhc2__mutmut_26": xǁRobustStandardErrorsǁhc2__mutmut_26,
        "xǁRobustStandardErrorsǁhc2__mutmut_27": xǁRobustStandardErrorsǁhc2__mutmut_27,
        "xǁRobustStandardErrorsǁhc2__mutmut_28": xǁRobustStandardErrorsǁhc2__mutmut_28,
        "xǁRobustStandardErrorsǁhc2__mutmut_29": xǁRobustStandardErrorsǁhc2__mutmut_29,
        "xǁRobustStandardErrorsǁhc2__mutmut_30": xǁRobustStandardErrorsǁhc2__mutmut_30,
        "xǁRobustStandardErrorsǁhc2__mutmut_31": xǁRobustStandardErrorsǁhc2__mutmut_31,
        "xǁRobustStandardErrorsǁhc2__mutmut_32": xǁRobustStandardErrorsǁhc2__mutmut_32,
        "xǁRobustStandardErrorsǁhc2__mutmut_33": xǁRobustStandardErrorsǁhc2__mutmut_33,
        "xǁRobustStandardErrorsǁhc2__mutmut_34": xǁRobustStandardErrorsǁhc2__mutmut_34,
    }
    xǁRobustStandardErrorsǁhc2__mutmut_orig.__name__ = "xǁRobustStandardErrorsǁhc2"

    def hc3(self) -> RobustCovarianceResult:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁRobustStandardErrorsǁhc3__mutmut_orig"),
            object.__getattribute__(self, "xǁRobustStandardErrorsǁhc3__mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_orig(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_1(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = None
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_2(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = None
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_3(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(None, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_4(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, None, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_5(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method=None, leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_6(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=None)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_7(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_8(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_9(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_10(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(
            self.X,
            self.resid,
            method="HC3",
        )
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_11(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="XXHC3XX", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_12(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="hc3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_13(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = None
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_14(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(None, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_15(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, None)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_16(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_17(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(
            self.bread,
        )
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_18(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = None

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_19(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(None)

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_20(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(None))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_21(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=None,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_22(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=None,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_23(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method=None,
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_24(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=None,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_25(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=None,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_26(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=None,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_27(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_28(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_29(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_30(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_31(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_32(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_33(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="XXHC3XX",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def xǁRobustStandardErrorsǁhc3__mutmut_34(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="hc3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    xǁRobustStandardErrorsǁhc3__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁRobustStandardErrorsǁhc3__mutmut_1": xǁRobustStandardErrorsǁhc3__mutmut_1,
        "xǁRobustStandardErrorsǁhc3__mutmut_2": xǁRobustStandardErrorsǁhc3__mutmut_2,
        "xǁRobustStandardErrorsǁhc3__mutmut_3": xǁRobustStandardErrorsǁhc3__mutmut_3,
        "xǁRobustStandardErrorsǁhc3__mutmut_4": xǁRobustStandardErrorsǁhc3__mutmut_4,
        "xǁRobustStandardErrorsǁhc3__mutmut_5": xǁRobustStandardErrorsǁhc3__mutmut_5,
        "xǁRobustStandardErrorsǁhc3__mutmut_6": xǁRobustStandardErrorsǁhc3__mutmut_6,
        "xǁRobustStandardErrorsǁhc3__mutmut_7": xǁRobustStandardErrorsǁhc3__mutmut_7,
        "xǁRobustStandardErrorsǁhc3__mutmut_8": xǁRobustStandardErrorsǁhc3__mutmut_8,
        "xǁRobustStandardErrorsǁhc3__mutmut_9": xǁRobustStandardErrorsǁhc3__mutmut_9,
        "xǁRobustStandardErrorsǁhc3__mutmut_10": xǁRobustStandardErrorsǁhc3__mutmut_10,
        "xǁRobustStandardErrorsǁhc3__mutmut_11": xǁRobustStandardErrorsǁhc3__mutmut_11,
        "xǁRobustStandardErrorsǁhc3__mutmut_12": xǁRobustStandardErrorsǁhc3__mutmut_12,
        "xǁRobustStandardErrorsǁhc3__mutmut_13": xǁRobustStandardErrorsǁhc3__mutmut_13,
        "xǁRobustStandardErrorsǁhc3__mutmut_14": xǁRobustStandardErrorsǁhc3__mutmut_14,
        "xǁRobustStandardErrorsǁhc3__mutmut_15": xǁRobustStandardErrorsǁhc3__mutmut_15,
        "xǁRobustStandardErrorsǁhc3__mutmut_16": xǁRobustStandardErrorsǁhc3__mutmut_16,
        "xǁRobustStandardErrorsǁhc3__mutmut_17": xǁRobustStandardErrorsǁhc3__mutmut_17,
        "xǁRobustStandardErrorsǁhc3__mutmut_18": xǁRobustStandardErrorsǁhc3__mutmut_18,
        "xǁRobustStandardErrorsǁhc3__mutmut_19": xǁRobustStandardErrorsǁhc3__mutmut_19,
        "xǁRobustStandardErrorsǁhc3__mutmut_20": xǁRobustStandardErrorsǁhc3__mutmut_20,
        "xǁRobustStandardErrorsǁhc3__mutmut_21": xǁRobustStandardErrorsǁhc3__mutmut_21,
        "xǁRobustStandardErrorsǁhc3__mutmut_22": xǁRobustStandardErrorsǁhc3__mutmut_22,
        "xǁRobustStandardErrorsǁhc3__mutmut_23": xǁRobustStandardErrorsǁhc3__mutmut_23,
        "xǁRobustStandardErrorsǁhc3__mutmut_24": xǁRobustStandardErrorsǁhc3__mutmut_24,
        "xǁRobustStandardErrorsǁhc3__mutmut_25": xǁRobustStandardErrorsǁhc3__mutmut_25,
        "xǁRobustStandardErrorsǁhc3__mutmut_26": xǁRobustStandardErrorsǁhc3__mutmut_26,
        "xǁRobustStandardErrorsǁhc3__mutmut_27": xǁRobustStandardErrorsǁhc3__mutmut_27,
        "xǁRobustStandardErrorsǁhc3__mutmut_28": xǁRobustStandardErrorsǁhc3__mutmut_28,
        "xǁRobustStandardErrorsǁhc3__mutmut_29": xǁRobustStandardErrorsǁhc3__mutmut_29,
        "xǁRobustStandardErrorsǁhc3__mutmut_30": xǁRobustStandardErrorsǁhc3__mutmut_30,
        "xǁRobustStandardErrorsǁhc3__mutmut_31": xǁRobustStandardErrorsǁhc3__mutmut_31,
        "xǁRobustStandardErrorsǁhc3__mutmut_32": xǁRobustStandardErrorsǁhc3__mutmut_32,
        "xǁRobustStandardErrorsǁhc3__mutmut_33": xǁRobustStandardErrorsǁhc3__mutmut_33,
        "xǁRobustStandardErrorsǁhc3__mutmut_34": xǁRobustStandardErrorsǁhc3__mutmut_34,
    }
    xǁRobustStandardErrorsǁhc3__mutmut_orig.__name__ = "xǁRobustStandardErrorsǁhc3"

    def compute(self, method: HC_TYPES = "HC1") -> RobustCovarianceResult:
        args = [method]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁRobustStandardErrorsǁcompute__mutmut_orig"),
            object.__getattribute__(self, "xǁRobustStandardErrorsǁcompute__mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁRobustStandardErrorsǁcompute__mutmut_orig(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_1(
        self, method: HC_TYPES = "XXHC1XX"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_2(
        self, method: HC_TYPES = "hc1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_3(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = None

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_4(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.lower()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_5(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper != "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_6(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "XXHC0XX":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_7(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "hc0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_8(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper != "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_9(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "XXHC1XX":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_10(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "hc1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_11(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper != "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_12(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "XXHC2XX":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_13(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "hc2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_14(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper != "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_15(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "XXHC3XX":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_16(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "hc3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )

    def xǁRobustStandardErrorsǁcompute__mutmut_17(
        self, method: HC_TYPES = "HC1"
    ) -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute("HC1")
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(None)

    xǁRobustStandardErrorsǁcompute__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁRobustStandardErrorsǁcompute__mutmut_1": xǁRobustStandardErrorsǁcompute__mutmut_1,
        "xǁRobustStandardErrorsǁcompute__mutmut_2": xǁRobustStandardErrorsǁcompute__mutmut_2,
        "xǁRobustStandardErrorsǁcompute__mutmut_3": xǁRobustStandardErrorsǁcompute__mutmut_3,
        "xǁRobustStandardErrorsǁcompute__mutmut_4": xǁRobustStandardErrorsǁcompute__mutmut_4,
        "xǁRobustStandardErrorsǁcompute__mutmut_5": xǁRobustStandardErrorsǁcompute__mutmut_5,
        "xǁRobustStandardErrorsǁcompute__mutmut_6": xǁRobustStandardErrorsǁcompute__mutmut_6,
        "xǁRobustStandardErrorsǁcompute__mutmut_7": xǁRobustStandardErrorsǁcompute__mutmut_7,
        "xǁRobustStandardErrorsǁcompute__mutmut_8": xǁRobustStandardErrorsǁcompute__mutmut_8,
        "xǁRobustStandardErrorsǁcompute__mutmut_9": xǁRobustStandardErrorsǁcompute__mutmut_9,
        "xǁRobustStandardErrorsǁcompute__mutmut_10": xǁRobustStandardErrorsǁcompute__mutmut_10,
        "xǁRobustStandardErrorsǁcompute__mutmut_11": xǁRobustStandardErrorsǁcompute__mutmut_11,
        "xǁRobustStandardErrorsǁcompute__mutmut_12": xǁRobustStandardErrorsǁcompute__mutmut_12,
        "xǁRobustStandardErrorsǁcompute__mutmut_13": xǁRobustStandardErrorsǁcompute__mutmut_13,
        "xǁRobustStandardErrorsǁcompute__mutmut_14": xǁRobustStandardErrorsǁcompute__mutmut_14,
        "xǁRobustStandardErrorsǁcompute__mutmut_15": xǁRobustStandardErrorsǁcompute__mutmut_15,
        "xǁRobustStandardErrorsǁcompute__mutmut_16": xǁRobustStandardErrorsǁcompute__mutmut_16,
        "xǁRobustStandardErrorsǁcompute__mutmut_17": xǁRobustStandardErrorsǁcompute__mutmut_17,
    }
    xǁRobustStandardErrorsǁcompute__mutmut_orig.__name__ = "xǁRobustStandardErrorsǁcompute"


def robust_covariance(
    X: np.ndarray, resid: np.ndarray, method: HC_TYPES = "HC1"
) -> RobustCovarianceResult:
    args = [X, resid, method]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_robust_covariance__mutmut_orig, x_robust_covariance__mutmut_mutants, args, kwargs, None
    )


def x_robust_covariance__mutmut_orig(
    X: np.ndarray, resid: np.ndarray, method: HC_TYPES = "HC1"
) -> RobustCovarianceResult:
    """
    Convenience function for computing robust covariance.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
        Type of heteroskedasticity-robust covariance

    Returns
    -------
    result : RobustCovarianceResult
        Covariance matrix and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import robust_covariance
    >>> result = robust_covariance(X, resid, method="HC1")
    >>> print(result.std_errors)
    """
    robust = RobustStandardErrors(X, resid)
    return robust.compute(method)


def x_robust_covariance__mutmut_1(
    X: np.ndarray, resid: np.ndarray, method: HC_TYPES = "XXHC1XX"
) -> RobustCovarianceResult:
    """
    Convenience function for computing robust covariance.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
        Type of heteroskedasticity-robust covariance

    Returns
    -------
    result : RobustCovarianceResult
        Covariance matrix and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import robust_covariance
    >>> result = robust_covariance(X, resid, method="HC1")
    >>> print(result.std_errors)
    """
    robust = RobustStandardErrors(X, resid)
    return robust.compute(method)


def x_robust_covariance__mutmut_2(
    X: np.ndarray, resid: np.ndarray, method: HC_TYPES = "hc1"
) -> RobustCovarianceResult:
    """
    Convenience function for computing robust covariance.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
        Type of heteroskedasticity-robust covariance

    Returns
    -------
    result : RobustCovarianceResult
        Covariance matrix and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import robust_covariance
    >>> result = robust_covariance(X, resid, method="HC1")
    >>> print(result.std_errors)
    """
    robust = RobustStandardErrors(X, resid)
    return robust.compute(method)


def x_robust_covariance__mutmut_3(
    X: np.ndarray, resid: np.ndarray, method: HC_TYPES = "HC1"
) -> RobustCovarianceResult:
    """
    Convenience function for computing robust covariance.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
        Type of heteroskedasticity-robust covariance

    Returns
    -------
    result : RobustCovarianceResult
        Covariance matrix and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import robust_covariance
    >>> result = robust_covariance(X, resid, method="HC1")
    >>> print(result.std_errors)
    """
    robust = None
    return robust.compute(method)


def x_robust_covariance__mutmut_4(
    X: np.ndarray, resid: np.ndarray, method: HC_TYPES = "HC1"
) -> RobustCovarianceResult:
    """
    Convenience function for computing robust covariance.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
        Type of heteroskedasticity-robust covariance

    Returns
    -------
    result : RobustCovarianceResult
        Covariance matrix and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import robust_covariance
    >>> result = robust_covariance(X, resid, method="HC1")
    >>> print(result.std_errors)
    """
    robust = RobustStandardErrors(None, resid)
    return robust.compute(method)


def x_robust_covariance__mutmut_5(
    X: np.ndarray, resid: np.ndarray, method: HC_TYPES = "HC1"
) -> RobustCovarianceResult:
    """
    Convenience function for computing robust covariance.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
        Type of heteroskedasticity-robust covariance

    Returns
    -------
    result : RobustCovarianceResult
        Covariance matrix and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import robust_covariance
    >>> result = robust_covariance(X, resid, method="HC1")
    >>> print(result.std_errors)
    """
    robust = RobustStandardErrors(X, None)
    return robust.compute(method)


def x_robust_covariance__mutmut_6(
    X: np.ndarray, resid: np.ndarray, method: HC_TYPES = "HC1"
) -> RobustCovarianceResult:
    """
    Convenience function for computing robust covariance.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
        Type of heteroskedasticity-robust covariance

    Returns
    -------
    result : RobustCovarianceResult
        Covariance matrix and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import robust_covariance
    >>> result = robust_covariance(X, resid, method="HC1")
    >>> print(result.std_errors)
    """
    robust = RobustStandardErrors(resid)
    return robust.compute(method)


def x_robust_covariance__mutmut_7(
    X: np.ndarray, resid: np.ndarray, method: HC_TYPES = "HC1"
) -> RobustCovarianceResult:
    """
    Convenience function for computing robust covariance.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
        Type of heteroskedasticity-robust covariance

    Returns
    -------
    result : RobustCovarianceResult
        Covariance matrix and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import robust_covariance
    >>> result = robust_covariance(X, resid, method="HC1")
    >>> print(result.std_errors)
    """
    robust = RobustStandardErrors(
        X,
    )
    return robust.compute(method)


def x_robust_covariance__mutmut_8(
    X: np.ndarray, resid: np.ndarray, method: HC_TYPES = "HC1"
) -> RobustCovarianceResult:
    """
    Convenience function for computing robust covariance.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
        Type of heteroskedasticity-robust covariance

    Returns
    -------
    result : RobustCovarianceResult
        Covariance matrix and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import robust_covariance
    >>> result = robust_covariance(X, resid, method="HC1")
    >>> print(result.std_errors)
    """
    robust = RobustStandardErrors(X, resid)
    return robust.compute(None)


x_robust_covariance__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_robust_covariance__mutmut_1": x_robust_covariance__mutmut_1,
    "x_robust_covariance__mutmut_2": x_robust_covariance__mutmut_2,
    "x_robust_covariance__mutmut_3": x_robust_covariance__mutmut_3,
    "x_robust_covariance__mutmut_4": x_robust_covariance__mutmut_4,
    "x_robust_covariance__mutmut_5": x_robust_covariance__mutmut_5,
    "x_robust_covariance__mutmut_6": x_robust_covariance__mutmut_6,
    "x_robust_covariance__mutmut_7": x_robust_covariance__mutmut_7,
    "x_robust_covariance__mutmut_8": x_robust_covariance__mutmut_8,
}
x_robust_covariance__mutmut_orig.__name__ = "x_robust_covariance"
