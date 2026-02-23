"""
Standard errors for Maximum Likelihood Estimation in panel models.

This module provides sandwich (robust) and cluster-robust standard errors
specifically for MLE estimators, including:
- Sandwich estimator (Huber-White robust SEs)
- Cluster-robust sandwich estimator
- Bootstrap standard errors
- Delta method for transformed parameters

References
----------
.. [1] White, H. (1982). "Maximum Likelihood Estimation of Misspecified Models."
       Econometrica, 50(1), 1-25.
.. [2] Cameron, A. C., & Trivedi, P. K. (2005). Microeconometrics.
       Cambridge University Press.
"""

from __future__ import annotations

import logging
from typing import Callable, Literal

import numpy as np

logger = logging.getLogger(__name__)
from typing import Annotated, ClassVar

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


class MLECovarianceResult:
    """
    Container for MLE covariance matrix estimation results.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix
    std_errors : np.ndarray
        Standard errors (sqrt of diagonal)
    method : str
        Method used ('nonrobust', 'robust', 'cluster', 'bootstrap')
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters

    Attributes
    ----------
    cov_matrix : np.ndarray
        Covariance matrix (k × k)
    std_errors : np.ndarray
        Standard errors (k,)
    method : str
        Estimation method
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters
    """

    def __init__(
        self,
        cov_matrix: np.ndarray,
        std_errors: np.ndarray,
        method: str,
        n_obs: int,
        n_params: int,
    ):
        args = [cov_matrix, std_errors, method, n_obs, n_params]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁMLECovarianceResultǁ__init____mutmut_orig"),
            object.__getattribute__(self, "xǁMLECovarianceResultǁ__init____mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁMLECovarianceResultǁ__init____mutmut_orig(
        self,
        cov_matrix: np.ndarray,
        std_errors: np.ndarray,
        method: str,
        n_obs: int,
        n_params: int,
    ):
        self.cov_matrix = cov_matrix
        self.std_errors = std_errors
        self.method = method
        self.n_obs = n_obs
        self.n_params = n_params

    def xǁMLECovarianceResultǁ__init____mutmut_1(
        self,
        cov_matrix: np.ndarray,
        std_errors: np.ndarray,
        method: str,
        n_obs: int,
        n_params: int,
    ):
        self.cov_matrix = None
        self.std_errors = std_errors
        self.method = method
        self.n_obs = n_obs
        self.n_params = n_params

    def xǁMLECovarianceResultǁ__init____mutmut_2(
        self,
        cov_matrix: np.ndarray,
        std_errors: np.ndarray,
        method: str,
        n_obs: int,
        n_params: int,
    ):
        self.cov_matrix = cov_matrix
        self.std_errors = None
        self.method = method
        self.n_obs = n_obs
        self.n_params = n_params

    def xǁMLECovarianceResultǁ__init____mutmut_3(
        self,
        cov_matrix: np.ndarray,
        std_errors: np.ndarray,
        method: str,
        n_obs: int,
        n_params: int,
    ):
        self.cov_matrix = cov_matrix
        self.std_errors = std_errors
        self.method = None
        self.n_obs = n_obs
        self.n_params = n_params

    def xǁMLECovarianceResultǁ__init____mutmut_4(
        self,
        cov_matrix: np.ndarray,
        std_errors: np.ndarray,
        method: str,
        n_obs: int,
        n_params: int,
    ):
        self.cov_matrix = cov_matrix
        self.std_errors = std_errors
        self.method = method
        self.n_obs = None
        self.n_params = n_params

    def xǁMLECovarianceResultǁ__init____mutmut_5(
        self,
        cov_matrix: np.ndarray,
        std_errors: np.ndarray,
        method: str,
        n_obs: int,
        n_params: int,
    ):
        self.cov_matrix = cov_matrix
        self.std_errors = std_errors
        self.method = method
        self.n_obs = n_obs
        self.n_params = None

    xǁMLECovarianceResultǁ__init____mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁMLECovarianceResultǁ__init____mutmut_1": xǁMLECovarianceResultǁ__init____mutmut_1,
        "xǁMLECovarianceResultǁ__init____mutmut_2": xǁMLECovarianceResultǁ__init____mutmut_2,
        "xǁMLECovarianceResultǁ__init____mutmut_3": xǁMLECovarianceResultǁ__init____mutmut_3,
        "xǁMLECovarianceResultǁ__init____mutmut_4": xǁMLECovarianceResultǁ__init____mutmut_4,
        "xǁMLECovarianceResultǁ__init____mutmut_5": xǁMLECovarianceResultǁ__init____mutmut_5,
    }
    xǁMLECovarianceResultǁ__init____mutmut_orig.__name__ = "xǁMLECovarianceResultǁ__init__"

    def __repr__(self) -> str:
        return (
            f"MLECovarianceResult("
            f"method='{self.method}', "
            f"n_obs={self.n_obs}, "
            f"n_params={self.n_params})"
        )


def sandwich_estimator(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    args = [hessian, scores, method]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_sandwich_estimator__mutmut_orig, x_sandwich_estimator__mutmut_mutants, args, kwargs, None
    )


def x_sandwich_estimator__mutmut_orig(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_1(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "XXrobustXX",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_2(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "ROBUST",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_3(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = None

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_4(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = None

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_5(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = +np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_6(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(None)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_7(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method != "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_8(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "XXnonrobustXX":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_9(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "NONROBUST":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_10(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = None

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_11(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method != "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_12(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "XXrobustXX":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_13(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "ROBUST":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_14(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = None

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_15(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = None

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_16(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(None)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_17(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = None

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_18(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(None)

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_19(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(None))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_20(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=None,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_21(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=None,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_22(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=None,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_23(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    _n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=None,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_24(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, _n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=None,
    )


def x_sandwich_estimator__mutmut_25(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_26(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_27(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_28(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    _n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_params=n_params,
    )


def x_sandwich_estimator__mutmut_29(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method="robust")
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, _n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
    )


x_sandwich_estimator__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_sandwich_estimator__mutmut_1": x_sandwich_estimator__mutmut_1,
    "x_sandwich_estimator__mutmut_2": x_sandwich_estimator__mutmut_2,
    "x_sandwich_estimator__mutmut_3": x_sandwich_estimator__mutmut_3,
    "x_sandwich_estimator__mutmut_4": x_sandwich_estimator__mutmut_4,
    "x_sandwich_estimator__mutmut_5": x_sandwich_estimator__mutmut_5,
    "x_sandwich_estimator__mutmut_6": x_sandwich_estimator__mutmut_6,
    "x_sandwich_estimator__mutmut_7": x_sandwich_estimator__mutmut_7,
    "x_sandwich_estimator__mutmut_8": x_sandwich_estimator__mutmut_8,
    "x_sandwich_estimator__mutmut_9": x_sandwich_estimator__mutmut_9,
    "x_sandwich_estimator__mutmut_10": x_sandwich_estimator__mutmut_10,
    "x_sandwich_estimator__mutmut_11": x_sandwich_estimator__mutmut_11,
    "x_sandwich_estimator__mutmut_12": x_sandwich_estimator__mutmut_12,
    "x_sandwich_estimator__mutmut_13": x_sandwich_estimator__mutmut_13,
    "x_sandwich_estimator__mutmut_14": x_sandwich_estimator__mutmut_14,
    "x_sandwich_estimator__mutmut_15": x_sandwich_estimator__mutmut_15,
    "x_sandwich_estimator__mutmut_16": x_sandwich_estimator__mutmut_16,
    "x_sandwich_estimator__mutmut_17": x_sandwich_estimator__mutmut_17,
    "x_sandwich_estimator__mutmut_18": x_sandwich_estimator__mutmut_18,
    "x_sandwich_estimator__mutmut_19": x_sandwich_estimator__mutmut_19,
    "x_sandwich_estimator__mutmut_20": x_sandwich_estimator__mutmut_20,
    "x_sandwich_estimator__mutmut_21": x_sandwich_estimator__mutmut_21,
    "x_sandwich_estimator__mutmut_22": x_sandwich_estimator__mutmut_22,
    "x_sandwich_estimator__mutmut_23": x_sandwich_estimator__mutmut_23,
    "x_sandwich_estimator__mutmut_24": x_sandwich_estimator__mutmut_24,
    "x_sandwich_estimator__mutmut_25": x_sandwich_estimator__mutmut_25,
    "x_sandwich_estimator__mutmut_26": x_sandwich_estimator__mutmut_26,
    "x_sandwich_estimator__mutmut_27": x_sandwich_estimator__mutmut_27,
    "x_sandwich_estimator__mutmut_28": x_sandwich_estimator__mutmut_28,
    "x_sandwich_estimator__mutmut_29": x_sandwich_estimator__mutmut_29,
}
x_sandwich_estimator__mutmut_orig.__name__ = "x_sandwich_estimator"


def cluster_robust_mle(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    args = [hessian, scores, cluster_ids, df_correction]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_cluster_robust_mle__mutmut_orig, x_cluster_robust_mle__mutmut_mutants, args, kwargs, None
    )


def x_cluster_robust_mle__mutmut_orig(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_1(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = False,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_2(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = None

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_3(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = None

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_4(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = +np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_5(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(None)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_6(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = None
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_7(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(None)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_8(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = None

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_9(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = None

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_10(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros(None)

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_11(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for _cluster in unique_clusters:
        mask = None
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_12(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids != cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_13(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for _cluster in unique_clusters:
        # Sum scores within cluster
        g_i = None
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_14(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=None)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_15(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=1)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_16(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat = np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_17(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat -= np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_18(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(None, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_19(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, None)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_20(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_21(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(
            g_i,
        )

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_22(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = None
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_23(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) / ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_24(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters * (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_25(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters + 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_26(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 2)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_27(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) * (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_28(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs + 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_29(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 2) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_30(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs + n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_31(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat = adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_32(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat /= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_33(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = None

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_34(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = None

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_35(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(None)

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_36(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(None))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_37(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=None,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_38(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=None,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_39(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=None,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_40(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=None,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_41(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=None,
    )


def x_cluster_robust_mle__mutmut_42(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_43(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_44(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_45(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_46(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
    )


def x_cluster_robust_mle__mutmut_47(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="XXclusterXX",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_cluster_robust_mle__mutmut_48(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="CLUSTER",
        n_obs=n_obs,
        n_params=n_params,
    )


x_cluster_robust_mle__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_cluster_robust_mle__mutmut_1": x_cluster_robust_mle__mutmut_1,
    "x_cluster_robust_mle__mutmut_2": x_cluster_robust_mle__mutmut_2,
    "x_cluster_robust_mle__mutmut_3": x_cluster_robust_mle__mutmut_3,
    "x_cluster_robust_mle__mutmut_4": x_cluster_robust_mle__mutmut_4,
    "x_cluster_robust_mle__mutmut_5": x_cluster_robust_mle__mutmut_5,
    "x_cluster_robust_mle__mutmut_6": x_cluster_robust_mle__mutmut_6,
    "x_cluster_robust_mle__mutmut_7": x_cluster_robust_mle__mutmut_7,
    "x_cluster_robust_mle__mutmut_8": x_cluster_robust_mle__mutmut_8,
    "x_cluster_robust_mle__mutmut_9": x_cluster_robust_mle__mutmut_9,
    "x_cluster_robust_mle__mutmut_10": x_cluster_robust_mle__mutmut_10,
    "x_cluster_robust_mle__mutmut_11": x_cluster_robust_mle__mutmut_11,
    "x_cluster_robust_mle__mutmut_12": x_cluster_robust_mle__mutmut_12,
    "x_cluster_robust_mle__mutmut_13": x_cluster_robust_mle__mutmut_13,
    "x_cluster_robust_mle__mutmut_14": x_cluster_robust_mle__mutmut_14,
    "x_cluster_robust_mle__mutmut_15": x_cluster_robust_mle__mutmut_15,
    "x_cluster_robust_mle__mutmut_16": x_cluster_robust_mle__mutmut_16,
    "x_cluster_robust_mle__mutmut_17": x_cluster_robust_mle__mutmut_17,
    "x_cluster_robust_mle__mutmut_18": x_cluster_robust_mle__mutmut_18,
    "x_cluster_robust_mle__mutmut_19": x_cluster_robust_mle__mutmut_19,
    "x_cluster_robust_mle__mutmut_20": x_cluster_robust_mle__mutmut_20,
    "x_cluster_robust_mle__mutmut_21": x_cluster_robust_mle__mutmut_21,
    "x_cluster_robust_mle__mutmut_22": x_cluster_robust_mle__mutmut_22,
    "x_cluster_robust_mle__mutmut_23": x_cluster_robust_mle__mutmut_23,
    "x_cluster_robust_mle__mutmut_24": x_cluster_robust_mle__mutmut_24,
    "x_cluster_robust_mle__mutmut_25": x_cluster_robust_mle__mutmut_25,
    "x_cluster_robust_mle__mutmut_26": x_cluster_robust_mle__mutmut_26,
    "x_cluster_robust_mle__mutmut_27": x_cluster_robust_mle__mutmut_27,
    "x_cluster_robust_mle__mutmut_28": x_cluster_robust_mle__mutmut_28,
    "x_cluster_robust_mle__mutmut_29": x_cluster_robust_mle__mutmut_29,
    "x_cluster_robust_mle__mutmut_30": x_cluster_robust_mle__mutmut_30,
    "x_cluster_robust_mle__mutmut_31": x_cluster_robust_mle__mutmut_31,
    "x_cluster_robust_mle__mutmut_32": x_cluster_robust_mle__mutmut_32,
    "x_cluster_robust_mle__mutmut_33": x_cluster_robust_mle__mutmut_33,
    "x_cluster_robust_mle__mutmut_34": x_cluster_robust_mle__mutmut_34,
    "x_cluster_robust_mle__mutmut_35": x_cluster_robust_mle__mutmut_35,
    "x_cluster_robust_mle__mutmut_36": x_cluster_robust_mle__mutmut_36,
    "x_cluster_robust_mle__mutmut_37": x_cluster_robust_mle__mutmut_37,
    "x_cluster_robust_mle__mutmut_38": x_cluster_robust_mle__mutmut_38,
    "x_cluster_robust_mle__mutmut_39": x_cluster_robust_mle__mutmut_39,
    "x_cluster_robust_mle__mutmut_40": x_cluster_robust_mle__mutmut_40,
    "x_cluster_robust_mle__mutmut_41": x_cluster_robust_mle__mutmut_41,
    "x_cluster_robust_mle__mutmut_42": x_cluster_robust_mle__mutmut_42,
    "x_cluster_robust_mle__mutmut_43": x_cluster_robust_mle__mutmut_43,
    "x_cluster_robust_mle__mutmut_44": x_cluster_robust_mle__mutmut_44,
    "x_cluster_robust_mle__mutmut_45": x_cluster_robust_mle__mutmut_45,
    "x_cluster_robust_mle__mutmut_46": x_cluster_robust_mle__mutmut_46,
    "x_cluster_robust_mle__mutmut_47": x_cluster_robust_mle__mutmut_47,
    "x_cluster_robust_mle__mutmut_48": x_cluster_robust_mle__mutmut_48,
}
x_cluster_robust_mle__mutmut_orig.__name__ = "x_cluster_robust_mle"


def delta_method(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    args = [vcov, transform_func, params, epsilon]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_delta_method__mutmut_orig, x_delta_method__mutmut_mutants, args, kwargs, None
    )


def x_delta_method__mutmut_orig(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_1(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1.0000001,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_2(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = None

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_3(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = None
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_4(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(None)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_5(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    transform_func(params)
    m = None

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_6(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = None

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_7(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros(None)

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_8(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(None):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_9(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = None
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_10(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = None
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_11(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] = epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_12(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] -= epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_13(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] = epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_14(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] += epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_15(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = None
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_16(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(None)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_17(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = None

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_18(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(None)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_19(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        transform_func(params_plus)
        transform_func(params_minus)

        jacobian[:, j] = None

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_20(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) * (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_21(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus + g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_22(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 / epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_23(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (3 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_24(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = None

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_25(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = None

    return vcov_transformed


def x_delta_method__mutmut_26(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) * 2

    return vcov_transformed


def x_delta_method__mutmut_27(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed - vcov_transformed.T) / 2

    return vcov_transformed


def x_delta_method__mutmut_28(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 3

    return vcov_transformed


x_delta_method__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_delta_method__mutmut_1": x_delta_method__mutmut_1,
    "x_delta_method__mutmut_2": x_delta_method__mutmut_2,
    "x_delta_method__mutmut_3": x_delta_method__mutmut_3,
    "x_delta_method__mutmut_4": x_delta_method__mutmut_4,
    "x_delta_method__mutmut_5": x_delta_method__mutmut_5,
    "x_delta_method__mutmut_6": x_delta_method__mutmut_6,
    "x_delta_method__mutmut_7": x_delta_method__mutmut_7,
    "x_delta_method__mutmut_8": x_delta_method__mutmut_8,
    "x_delta_method__mutmut_9": x_delta_method__mutmut_9,
    "x_delta_method__mutmut_10": x_delta_method__mutmut_10,
    "x_delta_method__mutmut_11": x_delta_method__mutmut_11,
    "x_delta_method__mutmut_12": x_delta_method__mutmut_12,
    "x_delta_method__mutmut_13": x_delta_method__mutmut_13,
    "x_delta_method__mutmut_14": x_delta_method__mutmut_14,
    "x_delta_method__mutmut_15": x_delta_method__mutmut_15,
    "x_delta_method__mutmut_16": x_delta_method__mutmut_16,
    "x_delta_method__mutmut_17": x_delta_method__mutmut_17,
    "x_delta_method__mutmut_18": x_delta_method__mutmut_18,
    "x_delta_method__mutmut_19": x_delta_method__mutmut_19,
    "x_delta_method__mutmut_20": x_delta_method__mutmut_20,
    "x_delta_method__mutmut_21": x_delta_method__mutmut_21,
    "x_delta_method__mutmut_22": x_delta_method__mutmut_22,
    "x_delta_method__mutmut_23": x_delta_method__mutmut_23,
    "x_delta_method__mutmut_24": x_delta_method__mutmut_24,
    "x_delta_method__mutmut_25": x_delta_method__mutmut_25,
    "x_delta_method__mutmut_26": x_delta_method__mutmut_26,
    "x_delta_method__mutmut_27": x_delta_method__mutmut_27,
    "x_delta_method__mutmut_28": x_delta_method__mutmut_28,
}
x_delta_method__mutmut_orig.__name__ = "x_delta_method"


def compute_mle_standard_errors(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    args = [model, params, se_type, entity_id]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_compute_mle_standard_errors__mutmut_orig,
        x_compute_mle_standard_errors__mutmut_mutants,
        args,
        kwargs,
        None,
    )


def x_compute_mle_standard_errors__mutmut_orig(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_1(
    model,
    params: np.ndarray,
    se_type: str = "XXclusterXX",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_2(
    model,
    params: np.ndarray,
    se_type: str = "CLUSTER",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_3(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = None

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_4(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(None)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_5(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type != "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_6(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "XXnonrobustXX":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_7(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "NONROBUST":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_8(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = None
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_9(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = +np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_10(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(None)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_11(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = None

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_12(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = +np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_13(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(None)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_14(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type not in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_15(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["XXrobustXX", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_16(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["ROBUST", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_17(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "XXclusterXX"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_18(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "CLUSTER"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_19(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = None

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_20(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(None)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_21(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type != "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_22(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "XXrobustXX":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_23(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "ROBUST":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_24(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = None
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_25(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(None, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_26(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, None, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_27(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method=None)
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_28(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_29(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_30(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(
                hessian,
                scores,
            )
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_31(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="XXrobustXX")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_32(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="ROBUST")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_33(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = None

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_34(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is not None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_35(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError(None)

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_36(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("XXentity_id required for cluster-robust SEsXX")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_37(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust ses")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_38(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("ENTITY_ID REQUIRED FOR CLUSTER-ROBUST SES")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_39(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = None
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_40(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(None, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_41(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, None, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_42(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, None)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_43(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_44(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_45(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(
                hessian,
                scores,
            )
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_46(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = None

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def x_compute_mle_standard_errors__mutmut_47(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(None)

    return cov_matrix


x_compute_mle_standard_errors__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_compute_mle_standard_errors__mutmut_1": x_compute_mle_standard_errors__mutmut_1,
    "x_compute_mle_standard_errors__mutmut_2": x_compute_mle_standard_errors__mutmut_2,
    "x_compute_mle_standard_errors__mutmut_3": x_compute_mle_standard_errors__mutmut_3,
    "x_compute_mle_standard_errors__mutmut_4": x_compute_mle_standard_errors__mutmut_4,
    "x_compute_mle_standard_errors__mutmut_5": x_compute_mle_standard_errors__mutmut_5,
    "x_compute_mle_standard_errors__mutmut_6": x_compute_mle_standard_errors__mutmut_6,
    "x_compute_mle_standard_errors__mutmut_7": x_compute_mle_standard_errors__mutmut_7,
    "x_compute_mle_standard_errors__mutmut_8": x_compute_mle_standard_errors__mutmut_8,
    "x_compute_mle_standard_errors__mutmut_9": x_compute_mle_standard_errors__mutmut_9,
    "x_compute_mle_standard_errors__mutmut_10": x_compute_mle_standard_errors__mutmut_10,
    "x_compute_mle_standard_errors__mutmut_11": x_compute_mle_standard_errors__mutmut_11,
    "x_compute_mle_standard_errors__mutmut_12": x_compute_mle_standard_errors__mutmut_12,
    "x_compute_mle_standard_errors__mutmut_13": x_compute_mle_standard_errors__mutmut_13,
    "x_compute_mle_standard_errors__mutmut_14": x_compute_mle_standard_errors__mutmut_14,
    "x_compute_mle_standard_errors__mutmut_15": x_compute_mle_standard_errors__mutmut_15,
    "x_compute_mle_standard_errors__mutmut_16": x_compute_mle_standard_errors__mutmut_16,
    "x_compute_mle_standard_errors__mutmut_17": x_compute_mle_standard_errors__mutmut_17,
    "x_compute_mle_standard_errors__mutmut_18": x_compute_mle_standard_errors__mutmut_18,
    "x_compute_mle_standard_errors__mutmut_19": x_compute_mle_standard_errors__mutmut_19,
    "x_compute_mle_standard_errors__mutmut_20": x_compute_mle_standard_errors__mutmut_20,
    "x_compute_mle_standard_errors__mutmut_21": x_compute_mle_standard_errors__mutmut_21,
    "x_compute_mle_standard_errors__mutmut_22": x_compute_mle_standard_errors__mutmut_22,
    "x_compute_mle_standard_errors__mutmut_23": x_compute_mle_standard_errors__mutmut_23,
    "x_compute_mle_standard_errors__mutmut_24": x_compute_mle_standard_errors__mutmut_24,
    "x_compute_mle_standard_errors__mutmut_25": x_compute_mle_standard_errors__mutmut_25,
    "x_compute_mle_standard_errors__mutmut_26": x_compute_mle_standard_errors__mutmut_26,
    "x_compute_mle_standard_errors__mutmut_27": x_compute_mle_standard_errors__mutmut_27,
    "x_compute_mle_standard_errors__mutmut_28": x_compute_mle_standard_errors__mutmut_28,
    "x_compute_mle_standard_errors__mutmut_29": x_compute_mle_standard_errors__mutmut_29,
    "x_compute_mle_standard_errors__mutmut_30": x_compute_mle_standard_errors__mutmut_30,
    "x_compute_mle_standard_errors__mutmut_31": x_compute_mle_standard_errors__mutmut_31,
    "x_compute_mle_standard_errors__mutmut_32": x_compute_mle_standard_errors__mutmut_32,
    "x_compute_mle_standard_errors__mutmut_33": x_compute_mle_standard_errors__mutmut_33,
    "x_compute_mle_standard_errors__mutmut_34": x_compute_mle_standard_errors__mutmut_34,
    "x_compute_mle_standard_errors__mutmut_35": x_compute_mle_standard_errors__mutmut_35,
    "x_compute_mle_standard_errors__mutmut_36": x_compute_mle_standard_errors__mutmut_36,
    "x_compute_mle_standard_errors__mutmut_37": x_compute_mle_standard_errors__mutmut_37,
    "x_compute_mle_standard_errors__mutmut_38": x_compute_mle_standard_errors__mutmut_38,
    "x_compute_mle_standard_errors__mutmut_39": x_compute_mle_standard_errors__mutmut_39,
    "x_compute_mle_standard_errors__mutmut_40": x_compute_mle_standard_errors__mutmut_40,
    "x_compute_mle_standard_errors__mutmut_41": x_compute_mle_standard_errors__mutmut_41,
    "x_compute_mle_standard_errors__mutmut_42": x_compute_mle_standard_errors__mutmut_42,
    "x_compute_mle_standard_errors__mutmut_43": x_compute_mle_standard_errors__mutmut_43,
    "x_compute_mle_standard_errors__mutmut_44": x_compute_mle_standard_errors__mutmut_44,
    "x_compute_mle_standard_errors__mutmut_45": x_compute_mle_standard_errors__mutmut_45,
    "x_compute_mle_standard_errors__mutmut_46": x_compute_mle_standard_errors__mutmut_46,
    "x_compute_mle_standard_errors__mutmut_47": x_compute_mle_standard_errors__mutmut_47,
}
x_compute_mle_standard_errors__mutmut_orig.__name__ = "x_compute_mle_standard_errors"


def bootstrap_mle(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    args = [estimate_func, y, X, n_bootstrap, cluster_ids, seed]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_bootstrap_mle__mutmut_orig, x_bootstrap_mle__mutmut_mutants, args, kwargs, None
    )


def x_bootstrap_mle__mutmut_orig(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_1(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 1000,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_2(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 43,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_3(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(None)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_4(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = None
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_5(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = None
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_6(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(None, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_7(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, None)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_8(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_9(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(
        y,
    )
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_10(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    estimate_func(y, X)
    n_params = None

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_11(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = None

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_12(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros(None)

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_13(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is not None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_14(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(None):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_15(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = None
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_16(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(None, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_17(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=None, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_18(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=None)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_19(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_20(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_21(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(
                n_obs,
                size=n_obs,
            )
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_22(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=False)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_23(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = None
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_24(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = None

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_25(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = None
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_26(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(None, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_27(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, None)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_28(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_29(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(
                    y_boot,
                )
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_30(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = None

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_31(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = None
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_32(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(None)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_33(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = None

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_34(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(None):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_35(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = None

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_36(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(None, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_37(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=None, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_38(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=None)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_39(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_40(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_41(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(
                unique_clusters,
                size=n_clusters,
            )

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_42(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=False)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_43(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = None
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_44(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for _cluster in sampled_clusters:
                indices.extend(None)

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_45(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for _cluster in sampled_clusters:
                indices.extend(np.where(None)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_46(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids != cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_47(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[1])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_48(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = None
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_49(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(None)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_50(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = None
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_51(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = None

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_52(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = None
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_53(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(None, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_54(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, None)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_55(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_56(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(
                    y_boot,
                )
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_57(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = None

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_58(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = None
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_59(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_60(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=None)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_61(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(None).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_62(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=2)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_63(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = None

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_64(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) <= n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_65(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap / 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_66(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 1.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_67(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            None,
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_68(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            None,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_69(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=None,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_70(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_71(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_72(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning, stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_73(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "XXMore than 50% of bootstrap replications failed. Results may be unreliable.XX",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_74(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "more than 50% of bootstrap replications failed. results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_75(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "MORE THAN 50% OF BOOTSTRAP REPLICATIONS FAILED. RESULTS MAY BE UNRELIABLE.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_76(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=3,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_77(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = None

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_78(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(None, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_79(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=None)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_80(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_81(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(
        bootstrap_estimates,
    )

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_82(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=True)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_83(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = None

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_84(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(None)

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_85(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(None))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_86(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=None,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_87(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=None,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_88(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=None,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_89(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=None,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_90(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=None,
    )


def x_bootstrap_mle__mutmut_91(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_92(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_93(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_94(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_95(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
    )


def x_bootstrap_mle__mutmut_96(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="XXbootstrapXX",
        n_obs=n_obs,
        n_params=n_params,
    )


def x_bootstrap_mle__mutmut_97(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except Exception:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            "More than 50% of bootstrap replications failed. Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="BOOTSTRAP",
        n_obs=n_obs,
        n_params=n_params,
    )


x_bootstrap_mle__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_bootstrap_mle__mutmut_1": x_bootstrap_mle__mutmut_1,
    "x_bootstrap_mle__mutmut_2": x_bootstrap_mle__mutmut_2,
    "x_bootstrap_mle__mutmut_3": x_bootstrap_mle__mutmut_3,
    "x_bootstrap_mle__mutmut_4": x_bootstrap_mle__mutmut_4,
    "x_bootstrap_mle__mutmut_5": x_bootstrap_mle__mutmut_5,
    "x_bootstrap_mle__mutmut_6": x_bootstrap_mle__mutmut_6,
    "x_bootstrap_mle__mutmut_7": x_bootstrap_mle__mutmut_7,
    "x_bootstrap_mle__mutmut_8": x_bootstrap_mle__mutmut_8,
    "x_bootstrap_mle__mutmut_9": x_bootstrap_mle__mutmut_9,
    "x_bootstrap_mle__mutmut_10": x_bootstrap_mle__mutmut_10,
    "x_bootstrap_mle__mutmut_11": x_bootstrap_mle__mutmut_11,
    "x_bootstrap_mle__mutmut_12": x_bootstrap_mle__mutmut_12,
    "x_bootstrap_mle__mutmut_13": x_bootstrap_mle__mutmut_13,
    "x_bootstrap_mle__mutmut_14": x_bootstrap_mle__mutmut_14,
    "x_bootstrap_mle__mutmut_15": x_bootstrap_mle__mutmut_15,
    "x_bootstrap_mle__mutmut_16": x_bootstrap_mle__mutmut_16,
    "x_bootstrap_mle__mutmut_17": x_bootstrap_mle__mutmut_17,
    "x_bootstrap_mle__mutmut_18": x_bootstrap_mle__mutmut_18,
    "x_bootstrap_mle__mutmut_19": x_bootstrap_mle__mutmut_19,
    "x_bootstrap_mle__mutmut_20": x_bootstrap_mle__mutmut_20,
    "x_bootstrap_mle__mutmut_21": x_bootstrap_mle__mutmut_21,
    "x_bootstrap_mle__mutmut_22": x_bootstrap_mle__mutmut_22,
    "x_bootstrap_mle__mutmut_23": x_bootstrap_mle__mutmut_23,
    "x_bootstrap_mle__mutmut_24": x_bootstrap_mle__mutmut_24,
    "x_bootstrap_mle__mutmut_25": x_bootstrap_mle__mutmut_25,
    "x_bootstrap_mle__mutmut_26": x_bootstrap_mle__mutmut_26,
    "x_bootstrap_mle__mutmut_27": x_bootstrap_mle__mutmut_27,
    "x_bootstrap_mle__mutmut_28": x_bootstrap_mle__mutmut_28,
    "x_bootstrap_mle__mutmut_29": x_bootstrap_mle__mutmut_29,
    "x_bootstrap_mle__mutmut_30": x_bootstrap_mle__mutmut_30,
    "x_bootstrap_mle__mutmut_31": x_bootstrap_mle__mutmut_31,
    "x_bootstrap_mle__mutmut_32": x_bootstrap_mle__mutmut_32,
    "x_bootstrap_mle__mutmut_33": x_bootstrap_mle__mutmut_33,
    "x_bootstrap_mle__mutmut_34": x_bootstrap_mle__mutmut_34,
    "x_bootstrap_mle__mutmut_35": x_bootstrap_mle__mutmut_35,
    "x_bootstrap_mle__mutmut_36": x_bootstrap_mle__mutmut_36,
    "x_bootstrap_mle__mutmut_37": x_bootstrap_mle__mutmut_37,
    "x_bootstrap_mle__mutmut_38": x_bootstrap_mle__mutmut_38,
    "x_bootstrap_mle__mutmut_39": x_bootstrap_mle__mutmut_39,
    "x_bootstrap_mle__mutmut_40": x_bootstrap_mle__mutmut_40,
    "x_bootstrap_mle__mutmut_41": x_bootstrap_mle__mutmut_41,
    "x_bootstrap_mle__mutmut_42": x_bootstrap_mle__mutmut_42,
    "x_bootstrap_mle__mutmut_43": x_bootstrap_mle__mutmut_43,
    "x_bootstrap_mle__mutmut_44": x_bootstrap_mle__mutmut_44,
    "x_bootstrap_mle__mutmut_45": x_bootstrap_mle__mutmut_45,
    "x_bootstrap_mle__mutmut_46": x_bootstrap_mle__mutmut_46,
    "x_bootstrap_mle__mutmut_47": x_bootstrap_mle__mutmut_47,
    "x_bootstrap_mle__mutmut_48": x_bootstrap_mle__mutmut_48,
    "x_bootstrap_mle__mutmut_49": x_bootstrap_mle__mutmut_49,
    "x_bootstrap_mle__mutmut_50": x_bootstrap_mle__mutmut_50,
    "x_bootstrap_mle__mutmut_51": x_bootstrap_mle__mutmut_51,
    "x_bootstrap_mle__mutmut_52": x_bootstrap_mle__mutmut_52,
    "x_bootstrap_mle__mutmut_53": x_bootstrap_mle__mutmut_53,
    "x_bootstrap_mle__mutmut_54": x_bootstrap_mle__mutmut_54,
    "x_bootstrap_mle__mutmut_55": x_bootstrap_mle__mutmut_55,
    "x_bootstrap_mle__mutmut_56": x_bootstrap_mle__mutmut_56,
    "x_bootstrap_mle__mutmut_57": x_bootstrap_mle__mutmut_57,
    "x_bootstrap_mle__mutmut_58": x_bootstrap_mle__mutmut_58,
    "x_bootstrap_mle__mutmut_59": x_bootstrap_mle__mutmut_59,
    "x_bootstrap_mle__mutmut_60": x_bootstrap_mle__mutmut_60,
    "x_bootstrap_mle__mutmut_61": x_bootstrap_mle__mutmut_61,
    "x_bootstrap_mle__mutmut_62": x_bootstrap_mle__mutmut_62,
    "x_bootstrap_mle__mutmut_63": x_bootstrap_mle__mutmut_63,
    "x_bootstrap_mle__mutmut_64": x_bootstrap_mle__mutmut_64,
    "x_bootstrap_mle__mutmut_65": x_bootstrap_mle__mutmut_65,
    "x_bootstrap_mle__mutmut_66": x_bootstrap_mle__mutmut_66,
    "x_bootstrap_mle__mutmut_67": x_bootstrap_mle__mutmut_67,
    "x_bootstrap_mle__mutmut_68": x_bootstrap_mle__mutmut_68,
    "x_bootstrap_mle__mutmut_69": x_bootstrap_mle__mutmut_69,
    "x_bootstrap_mle__mutmut_70": x_bootstrap_mle__mutmut_70,
    "x_bootstrap_mle__mutmut_71": x_bootstrap_mle__mutmut_71,
    "x_bootstrap_mle__mutmut_72": x_bootstrap_mle__mutmut_72,
    "x_bootstrap_mle__mutmut_73": x_bootstrap_mle__mutmut_73,
    "x_bootstrap_mle__mutmut_74": x_bootstrap_mle__mutmut_74,
    "x_bootstrap_mle__mutmut_75": x_bootstrap_mle__mutmut_75,
    "x_bootstrap_mle__mutmut_76": x_bootstrap_mle__mutmut_76,
    "x_bootstrap_mle__mutmut_77": x_bootstrap_mle__mutmut_77,
    "x_bootstrap_mle__mutmut_78": x_bootstrap_mle__mutmut_78,
    "x_bootstrap_mle__mutmut_79": x_bootstrap_mle__mutmut_79,
    "x_bootstrap_mle__mutmut_80": x_bootstrap_mle__mutmut_80,
    "x_bootstrap_mle__mutmut_81": x_bootstrap_mle__mutmut_81,
    "x_bootstrap_mle__mutmut_82": x_bootstrap_mle__mutmut_82,
    "x_bootstrap_mle__mutmut_83": x_bootstrap_mle__mutmut_83,
    "x_bootstrap_mle__mutmut_84": x_bootstrap_mle__mutmut_84,
    "x_bootstrap_mle__mutmut_85": x_bootstrap_mle__mutmut_85,
    "x_bootstrap_mle__mutmut_86": x_bootstrap_mle__mutmut_86,
    "x_bootstrap_mle__mutmut_87": x_bootstrap_mle__mutmut_87,
    "x_bootstrap_mle__mutmut_88": x_bootstrap_mle__mutmut_88,
    "x_bootstrap_mle__mutmut_89": x_bootstrap_mle__mutmut_89,
    "x_bootstrap_mle__mutmut_90": x_bootstrap_mle__mutmut_90,
    "x_bootstrap_mle__mutmut_91": x_bootstrap_mle__mutmut_91,
    "x_bootstrap_mle__mutmut_92": x_bootstrap_mle__mutmut_92,
    "x_bootstrap_mle__mutmut_93": x_bootstrap_mle__mutmut_93,
    "x_bootstrap_mle__mutmut_94": x_bootstrap_mle__mutmut_94,
    "x_bootstrap_mle__mutmut_95": x_bootstrap_mle__mutmut_95,
    "x_bootstrap_mle__mutmut_96": x_bootstrap_mle__mutmut_96,
    "x_bootstrap_mle__mutmut_97": x_bootstrap_mle__mutmut_97,
}
x_bootstrap_mle__mutmut_orig.__name__ = "x_bootstrap_mle"
