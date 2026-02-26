"""
Lagrange Multiplier (LM) tests for spatial dependence.

Based on Anselin (1988) and Anselin et al. (1996).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from panelbox.core.spatial_weights import SpatialWeights

from ..base import ValidationTest, ValidationTestResult
from .utils import validate_spatial_weights

logger = logging.getLogger(__name__)


class LMLagTest(ValidationTest):
    """
    LM test for spatial lag dependence (Anselin 1988).

    Tests H0: ρ=0 in the spatial lag model:
    y = ρWy + Xβ + ε

    This test requires only OLS residuals and does not need
    estimation of the spatial model.

    Parameters
    ----------
    ols_result : object
        OLS regression result with attributes:
        - resid: residuals
        - fittedvalues: fitted values
        - nobs: number of observations
    W : np.ndarray
        Spatial weight matrix

    References
    ----------
    Anselin, L. (1988). Lagrange multiplier test diagnostics for
    spatial dependence and spatial heterogeneity. Geographical
    Analysis, 20(1), 1-17.
    """

    def __init__(self, ols_result, W: np.ndarray | SpatialWeights):
        """Initialize LM-lag test."""
        self.ols_result = ols_result
        self.W = validate_spatial_weights(W)

        # Extract key components
        self.e_hat = np.asarray(ols_result.resid).flatten()
        self.y_hat = np.asarray(ols_result.fittedvalues).flatten()
        self.n = len(self.e_hat)

        # Validate dimensions
        if self.W.shape[0] != self.n:
            # Handle panel data case - might need to extract cross-section
            if hasattr(ols_result, "model") and hasattr(ols_result.model, "N"):
                self.N = ols_result.model.N
                self.T = ols_result.model.T
                if self.W.shape[0] == self.N:
                    # W is for cross-section, need to expand for panel
                    self._setup_panel_weights()
                else:
                    raise ValueError(f"W dimension {self.W.shape[0]} incompatible with N={self.N}")
            else:
                raise ValueError(f"W dimension {self.W.shape[0]} != n={self.n}")

    def _setup_panel_weights(self):
        """Setup spatial weights for panel data."""
        # For panel data, use block diagonal W matrix
        # W_panel = I_T ⊗ W
        from scipy.linalg import block_diag

        W_blocks = [self.W for _ in range(self.T)]
        self.W_full = block_diag(*W_blocks)
        self.W = self.W_full

    def run(self, alpha: float = 0.05) -> ValidationTestResult:
        """
        Compute LM-lag statistic.

        Parameters
        ----------
        alpha : float
            Significance level

        Returns
        -------
        ValidationTestResult
            Test results
        """
        # Compute test components
        sigma2 = (self.e_hat @ self.e_hat) / self.n

        # Spatial lag of fitted values
        if hasattr(self.ols_result, "model") and hasattr(self.ols_result.model, "data"):
            # Panel data: need to properly compute spatial lag
            y = self.ols_result.model.data.y
            Wy = self.W @ y
        else:
            # Cross-section: use fitted values as proxy
            y = self.y_hat + self.e_hat  # Reconstruct y
            Wy = self.W @ y

        # Numerator: (ê'Wy / σ²)²
        numerator = (self.e_hat @ Wy) ** 2 / sigma2**2

        # Trace terms for denominator
        WW = self.W @ self.W
        WtW = self.W.T @ self.W
        tr_WtW_WW = np.trace(WtW + WW)

        # Additional term for denominator (involves (WXβ)'M(WXβ))
        # Simplified version using fitted values
        WXb = self.W @ self.y_hat
        M = np.eye(self.n) - np.ones((self.n, 1)) @ np.ones((1, self.n)) / self.n
        WXb_M_WXb = WXb @ M @ WXb

        # Denominator
        T_term = tr_WtW_WW + (WXb_M_WXb / sigma2)

        # LM statistic
        lm_stat = numerator / T_term

        # P-value from χ²(1)
        pvalue = 1 - stats.chi2.cdf(lm_stat, df=1)

        # Interpret results
        if pvalue < alpha:
            conclusion = "Reject H0: Spatial lag dependence detected (consider SAR model)"
        else:
            conclusion = "Fail to reject H0: No significant spatial lag dependence"

        return ValidationTestResult(
            test_name="LM-lag",
            statistic=float(lm_stat),
            pvalue=float(pvalue),
            null_hypothesis="No spatial lag dependence (ρ=0)",
            alternative_hypothesis="Spatial lag dependence present (ρ≠0)",
            alpha=alpha,
            df=1,
            metadata={
                "distribution": "χ²(1)",
                "sigma2": float(sigma2),
                "trace_term": float(tr_WtW_WW),
                "conclusion": conclusion,
            },
        )


class LMErrorTest(ValidationTest):
    """
    LM test for spatial error dependence (Anselin 1988).

    Tests H0: λ=0 in the spatial error model:
    y = Xβ + u, where u = λWu + ε

    Parameters
    ----------
    ols_result : object
        OLS regression result
    W : np.ndarray
        Spatial weight matrix
    """

    def __init__(self, ols_result, W: np.ndarray | SpatialWeights):
        """Initialize LM-error test."""
        self.ols_result = ols_result
        self.W = validate_spatial_weights(W)

        # Extract residuals
        self.e_hat = np.asarray(ols_result.resid).flatten()
        self.n = len(self.e_hat)

        # Validate and setup for panel if needed
        if self.W.shape[0] != self.n:
            if hasattr(ols_result, "model") and hasattr(ols_result.model, "N"):
                self.N = ols_result.model.N
                self.T = ols_result.model.T
                if self.W.shape[0] == self.N:
                    self._setup_panel_weights()
                else:
                    raise ValueError(f"W dimension {self.W.shape[0]} incompatible")
            else:
                raise ValueError(f"W dimension {self.W.shape[0]} != n={self.n}")

    def _setup_panel_weights(self):
        """Setup spatial weights for panel data."""
        from scipy.linalg import block_diag

        W_blocks = [self.W for _ in range(self.T)]
        self.W_full = block_diag(*W_blocks)
        self.W = self.W_full

    def run(self, alpha: float = 0.05) -> ValidationTestResult:
        """
        Compute LM-error statistic.

        Parameters
        ----------
        alpha : float
            Significance level

        Returns
        -------
        ValidationTestResult
            Test results
        """
        # Compute test components
        sigma2 = (self.e_hat @ self.e_hat) / self.n

        # Spatial lag of residuals
        We_hat = self.W @ self.e_hat

        # Numerator: (ê'Wê / σ²)²
        numerator = (self.e_hat @ We_hat) ** 2 / sigma2**2

        # Trace term
        WW = self.W @ self.W
        WtW = self.W.T @ self.W
        tr_WtW_WW = np.trace(WtW + WW)

        # LM statistic
        lm_stat = numerator / tr_WtW_WW

        # P-value from χ²(1)
        pvalue = 1 - stats.chi2.cdf(lm_stat, df=1)

        # Interpret results
        if pvalue < alpha:
            conclusion = "Reject H0: Spatial error dependence detected (consider SEM model)"
        else:
            conclusion = "Fail to reject H0: No significant spatial error dependence"

        return ValidationTestResult(
            test_name="LM-error",
            statistic=float(lm_stat),
            pvalue=float(pvalue),
            null_hypothesis="No spatial error dependence (λ=0)",
            alternative_hypothesis="Spatial error dependence present (λ≠0)",
            alpha=alpha,
            df=1,
            metadata={
                "distribution": "χ²(1)",
                "sigma2": float(sigma2),
                "trace_term": float(tr_WtW_WW),
                "conclusion": conclusion,
            },
        )


class RobustLMLagTest(ValidationTest):
    """
    Robust LM test for spatial lag (Anselin et al. 1996).

    Robust to the presence of spatial error dependence.

    Parameters
    ----------
    ols_result : object
        OLS regression result
    W : np.ndarray
        Spatial weight matrix

    References
    ----------
    Anselin, L., Bera, A. K., Florax, R., & Yoon, M. J. (1996).
    Simple diagnostic tests for spatial dependence. Regional
    Science and Urban Economics, 26(1), 77-104.
    """

    def __init__(self, ols_result, W: np.ndarray | SpatialWeights):
        """Initialize Robust LM-lag test."""
        self.ols_result = ols_result
        self.W = validate_spatial_weights(W)
        self.e_hat = np.asarray(ols_result.resid).flatten()
        self.y_hat = np.asarray(ols_result.fittedvalues).flatten()
        self.n = len(self.e_hat)

        # Setup for panel if needed
        if self.W.shape[0] != self.n:
            if hasattr(ols_result, "model") and hasattr(ols_result.model, "N"):
                self.N = ols_result.model.N
                self.T = ols_result.model.T
                if self.W.shape[0] == self.N:
                    self._setup_panel_weights()

    def _setup_panel_weights(self):
        """Setup spatial weights for panel data."""
        from scipy.linalg import block_diag

        W_blocks = [self.W for _ in range(self.T)]
        self.W_full = block_diag(*W_blocks)
        self.W = self.W_full

    def run(self, alpha: float = 0.05) -> ValidationTestResult:
        """
        Compute Robust LM-lag statistic.

        The robust version adjusts for possible spatial error.
        """
        sigma2 = (self.e_hat @ self.e_hat) / self.n

        # Compute components
        if hasattr(self.ols_result, "model") and hasattr(self.ols_result.model, "data"):
            y = self.ols_result.model.data.y
        else:
            y = self.y_hat + self.e_hat

        Wy = self.W @ y
        We = self.W @ self.e_hat

        # Trace terms
        WW = self.W @ self.W
        WtW = self.W.T @ self.W
        tr_WtW_WW = np.trace(WtW + WW)
        tr_W2 = np.trace(WW)

        # Components for robust test
        d_lag = (self.e_hat @ Wy) / sigma2
        d_err = (self.e_hat @ We) / sigma2

        # Robust LM-lag statistic
        numerator = (d_lag - (tr_W2 / tr_WtW_WW) * d_err) ** 2

        # Additional terms for denominator
        WXb = self.W @ self.y_hat
        M = np.eye(self.n) - np.ones((self.n, 1)) @ np.ones((1, self.n)) / self.n
        WXb_M_WXb = WXb @ M @ WXb / sigma2

        T1 = WXb_M_WXb + tr_WtW_WW
        T2 = tr_W2**2 / tr_WtW_WW

        denominator = T1 - T2

        # Statistic
        rlm_stat = numerator / denominator if denominator > 0 else 0

        # P-value from χ²(1)
        pvalue = 1 - stats.chi2.cdf(rlm_stat, df=1)

        # Interpret results
        if pvalue < alpha:
            conclusion = "Reject H0: Spatial lag dependence detected (robust to error)"
        else:
            conclusion = "Fail to reject H0: No significant spatial lag dependence"

        return ValidationTestResult(
            test_name="Robust LM-lag",
            statistic=float(rlm_stat),
            pvalue=float(pvalue),
            null_hypothesis="No spatial lag dependence (ρ=0), robust to error",
            alternative_hypothesis="Spatial lag dependence present (ρ≠0)",
            alpha=alpha,
            df=1,
            metadata={
                "distribution": "χ²(1)",
                "sigma2": float(sigma2),
                "trace_WtW_WW": float(tr_WtW_WW),
                "trace_W2": float(tr_W2),
                "conclusion": conclusion,
            },
        )


class RobustLMErrorTest(ValidationTest):
    """
    Robust LM test for spatial error (Anselin et al. 1996).

    Robust to the presence of spatial lag dependence.

    Parameters
    ----------
    ols_result : object
        OLS regression result
    W : np.ndarray
        Spatial weight matrix
    """

    def __init__(self, ols_result, W: np.ndarray | SpatialWeights):
        """Initialize Robust LM-error test."""
        self.ols_result = ols_result
        self.W = validate_spatial_weights(W)
        self.e_hat = np.asarray(ols_result.resid).flatten()
        self.y_hat = np.asarray(ols_result.fittedvalues).flatten()
        self.n = len(self.e_hat)

        # Setup for panel if needed
        if self.W.shape[0] != self.n:
            if hasattr(ols_result, "model") and hasattr(ols_result.model, "N"):
                self.N = ols_result.model.N
                self.T = ols_result.model.T
                if self.W.shape[0] == self.N:
                    self._setup_panel_weights()

    def _setup_panel_weights(self):
        """Setup spatial weights for panel data."""
        from scipy.linalg import block_diag

        W_blocks = [self.W for _ in range(self.T)]
        self.W_full = block_diag(*W_blocks)
        self.W = self.W_full

    def run(self, alpha: float = 0.05) -> ValidationTestResult:
        """
        Compute Robust LM-error statistic.

        The robust version adjusts for possible spatial lag.
        """
        sigma2 = (self.e_hat @ self.e_hat) / self.n

        # Compute components
        if hasattr(self.ols_result, "model") and hasattr(self.ols_result.model, "data"):
            y = self.ols_result.model.data.y
        else:
            y = self.y_hat + self.e_hat

        Wy = self.W @ y
        We = self.W @ self.e_hat

        # Trace terms
        WW = self.W @ self.W
        WtW = self.W.T @ self.W
        tr_WtW_WW = np.trace(WtW + WW)

        # Components
        d_lag = (self.e_hat @ Wy) / sigma2
        d_err = (self.e_hat @ We) / sigma2

        # T term (from WXβ)
        WXb = self.W @ self.y_hat
        M = np.eye(self.n) - np.ones((self.n, 1)) @ np.ones((1, self.n)) / self.n
        T = (WXb @ M @ WXb) / sigma2

        # Robust LM-error statistic
        numerator = (d_err - d_lag / (T + tr_WtW_WW)) ** 2

        denominator = tr_WtW_WW * (1 - T / (T + tr_WtW_WW))

        # Statistic
        rlm_stat = numerator / denominator if denominator > 0 else 0

        # P-value from χ²(1)
        pvalue = 1 - stats.chi2.cdf(rlm_stat, df=1)

        # Interpret results
        if pvalue < alpha:
            conclusion = "Reject H0: Spatial error dependence detected (robust to lag)"
        else:
            conclusion = "Fail to reject H0: No significant spatial error dependence"

        return ValidationTestResult(
            test_name="Robust LM-error",
            statistic=float(rlm_stat),
            pvalue=float(pvalue),
            null_hypothesis="No spatial error dependence (λ=0), robust to lag",
            alternative_hypothesis="Spatial error dependence present (λ≠0)",
            alpha=alpha,
            df=1,
            metadata={
                "distribution": "χ²(1)",
                "sigma2": float(sigma2),
                "trace_term": float(tr_WtW_WW),
                "T_term": float(T),
                "conclusion": conclusion,
            },
        )


def run_lm_tests(
    ols_result, W: np.ndarray | SpatialWeights, alpha: float = 0.05, verbose: bool = True
) -> dict:
    """
    Run complete battery of LM tests for spatial dependence.

    Runs all four LM tests (lag, error, robust lag, robust error)
    and provides a model recommendation based on the results.

    Parameters
    ----------
    ols_result : object
        OLS regression result
    W : np.ndarray or SpatialWeights
        Spatial weight matrix
    alpha : float
        Significance level
    verbose : bool
        Whether to print results

    Returns
    -------
    dict
        Dictionary with test results and recommendation:
        - 'lm_lag': LMLagTest result
        - 'lm_error': LMErrorTest result
        - 'robust_lm_lag': RobustLMLagTest result
        - 'robust_lm_error': RobustLMErrorTest result
        - 'recommendation': str ('SAR', 'SEM', 'SDM', 'OLS')
        - 'summary': pd.DataFrame with all results

    References
    ----------
    Decision rule based on Anselin and Florax (1995):
    1. If neither LM test is significant → OLS
    2. If only one is significant → corresponding model
    3. If both are significant → check robust tests
    4. If both robust are significant → consider SDM
    """
    # Run all tests
    lm_lag = LMLagTest(ols_result, W).run(alpha)
    lm_error = LMErrorTest(ols_result, W).run(alpha)
    robust_lm_lag = RobustLMLagTest(ols_result, W).run(alpha)
    robust_lm_error = RobustLMErrorTest(ols_result, W).run(alpha)

    # Decision rule for model selection
    lag_sig = lm_lag.pvalue < alpha
    error_sig = lm_error.pvalue < alpha
    robust_lag_sig = robust_lm_lag.pvalue < alpha
    robust_error_sig = robust_lm_error.pvalue < alpha

    if not lag_sig and not error_sig:
        # Neither significant
        recommendation = "OLS"
        reason = "No spatial dependence detected"
    elif lag_sig and not error_sig:
        # Only lag significant
        recommendation = "SAR"
        reason = "Spatial lag dependence detected"
    elif error_sig and not lag_sig:
        # Only error significant
        recommendation = "SEM"
        reason = "Spatial error dependence detected"
    else:
        # Both significant - check robust tests
        if robust_lag_sig and not robust_error_sig:
            recommendation = "SAR"
            reason = "Spatial lag dependence (robust test)"
        elif robust_error_sig and not robust_lag_sig:
            recommendation = "SEM"
            reason = "Spatial error dependence (robust test)"
        elif robust_lag_sig and robust_error_sig:
            # Both robust tests significant
            recommendation = "SDM"
            reason = "Both lag and error dependence detected (consider SDM or SARAR)"
        else:
            # Neither robust test significant (rare)
            recommendation = "OLS"
            reason = "Dependence not confirmed by robust tests"

    # Create summary DataFrame
    summary_data = {
        "Test": ["LM-lag", "LM-error", "Robust LM-lag", "Robust LM-error"],
        "Statistic": [
            lm_lag.statistic,
            lm_error.statistic,
            robust_lm_lag.statistic,
            robust_lm_error.statistic,
        ],
        "p-value": [lm_lag.pvalue, lm_error.pvalue, robust_lm_lag.pvalue, robust_lm_error.pvalue],
        "Significant": [lag_sig, error_sig, robust_lag_sig, robust_error_sig],
    }

    summary = pd.DataFrame(summary_data)

    # Print results if verbose
    if verbose:
        logger.info("=" * 50)
        logger.info("LM TESTS FOR SPATIAL DEPENDENCE")
        logger.info("=" * 50)
        logger.info(summary.to_string(index=False))
        logger.info("-" * 50)
        logger.info(f"Recommendation: {recommendation}")
        logger.info(f"Reason: {reason}")
        logger.info("=" * 50)

    return {
        "lm_lag": lm_lag,
        "lm_error": lm_error,
        "robust_lm_lag": robust_lm_lag,
        "robust_lm_error": robust_lm_error,
        "recommendation": recommendation,
        "reason": reason,
        "summary": summary,
    }
