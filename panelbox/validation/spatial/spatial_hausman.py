"""Spatial Hausman test for model specification."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from ..base import ValidationTest, ValidationTestResult

logger = logging.getLogger(__name__)


class SpatialHausmanTest(ValidationTest):
    """
    Hausman test for spatial model specification.

    Compares parameter estimates from two spatial models to test
    whether both are consistent or only one is consistent.

    The test is based on the principle that under H0, both estimators
    are consistent but one is more efficient. Under Ha, only one
    estimator is consistent.

    Common comparisons:
    - SAR vs OLS: Test if spatial lag is needed
    - SAR vs SEM: Test which spatial structure is appropriate
    - FE vs RE: Test fixed vs random effects (spatial context)

    Parameters
    ----------
    result1 : object
        First model result (typically the efficient estimator under H0)
        Must have attributes: params, cov_params() or bse
    result2 : object
        Second model result (typically the consistent estimator)
        Must have attributes: params, cov_params() or bse

    Attributes
    ----------
    result1 : object
        First model result
    result2 : object
        Second model result
    common_params : list
        Parameter names common to both models

    References
    ----------
    Hausman, J. A. (1978). Specification tests in econometrics.
    Econometrica, 46(6), 1251-1271.

    Mutl, J., & Pfaffermayr, M. (2011). The Hausman test in a
    Cliff and Ord panel model. The Econometrics Journal, 14(1), 48-76.
    """

    def __init__(self, result1, result2):
        """
        Initialize Spatial Hausman test.

        Parameters
        ----------
        result1 : object
            First model result
        result2 : object
            Second model result
        """
        self.result1 = result1
        self.result2 = result2

        # Extract parameter names
        self._extract_parameters()

    def _extract_parameters(self):
        """Extract and match parameters from both models."""
        # Get parameter names
        if hasattr(self.result1, "params"):
            if isinstance(self.result1.params, pd.Series):
                params1_names = list(self.result1.params.index)
            else:
                params1_names = [f"beta_{i}" for i in range(len(self.result1.params))]
        else:
            raise ValueError("result1 must have 'params' attribute")

        if hasattr(self.result2, "params"):
            if isinstance(self.result2.params, pd.Series):
                params2_names = list(self.result2.params.index)
            else:
                params2_names = [f"beta_{i}" for i in range(len(self.result2.params))]
        else:
            raise ValueError("result2 must have 'params' attribute")

        # Find common parameters (exclude spatial parameters)
        spatial_params = ["rho", "lambda", "theta", "spatial_lag", "spatial_error"]

        params1_clean = [p for p in params1_names if p.lower() not in spatial_params]
        params2_clean = [p for p in params2_names if p.lower() not in spatial_params]

        # Find intersection
        self.common_params = list(set(params1_clean) & set(params2_clean))

        if len(self.common_params) == 0:
            # Try positional matching
            min_len = min(len(params1_clean), len(params2_clean))
            self.common_params = list(range(min_len))
            self.use_positional = True
        else:
            self.common_params.sort()
            self.use_positional = False

    def run(
        self, alpha: float = 0.05, subset: list | None = None, robust: bool = False
    ) -> ValidationTestResult:
        """
        Compute Hausman test statistic.

        Parameters
        ----------
        alpha : float
            Significance level
        subset : list, optional
            Subset of parameters to test
        robust : bool
            Whether to use robust variance estimation

        Returns
        -------
        ValidationTestResult
            Test results
        """
        # Extract parameters
        if subset is not None:
            test_params = subset
        else:
            test_params = self.common_params

        # Get parameter estimates
        beta1 = self._get_params(self.result1, test_params)
        beta2 = self._get_params(self.result2, test_params)

        # Check dimensions
        if len(beta1) != len(beta2):
            raise ValueError("Parameter vectors must have same length")

        # Compute difference
        diff = beta1 - beta2

        # Get covariance matrices
        V1 = self._get_covariance(self.result1, test_params)
        V2 = self._get_covariance(self.result2, test_params)

        # Compute variance of difference
        # V(b1 - b2) = V(b2) - V(b1) under H0
        # Note: More efficient estimator has smaller variance
        V_diff = V2 - V1

        # Check if V_diff is positive definite
        try:
            # Add small regularization if needed
            eigenvalues = np.linalg.eigvalsh(V_diff)
            if np.min(eigenvalues) < 1e-10:
                V_diff = V_diff + np.eye(len(diff)) * 1e-8

            # Compute test statistic
            V_diff_inv = np.linalg.inv(V_diff)
            H_stat = diff @ V_diff_inv @ diff

        except np.linalg.LinAlgError:
            # Matrix not invertible
            # Use generalized inverse
            V_diff_pinv = np.linalg.pinv(V_diff)
            H_stat = diff @ V_diff_pinv @ diff

        # Degrees of freedom
        K = len(diff)

        # P-value from χ²(K)
        pvalue = 1 - stats.chi2.cdf(H_stat, df=K)

        # Interpret results
        if pvalue < alpha:
            conclusion = (
                f"Reject H0 at {alpha:.0%} level: "
                f"Significant differences between models. "
                f"Prefer the consistent estimator."
            )
        else:
            conclusion = (
                f"Fail to reject H0 at {alpha:.0%} level: "
                f"Both models appear consistent. "
                f"Can use the more efficient estimator."
            )

        # Determine which model to prefer
        model1_name = self._get_model_name(self.result1)
        model2_name = self._get_model_name(self.result2)

        return ValidationTestResult(
            test_name="Spatial Hausman Test",
            statistic=float(H_stat),
            pvalue=float(pvalue),
            null_hypothesis="Both models are consistent",
            alternative_hypothesis="Only one model is consistent",
            alpha=alpha,
            df=K,
            metadata={
                "distribution": f"χ²({K})",
                "degrees_of_freedom": K,
                "model1": model1_name,
                "model2": model2_name,
                "n_parameters_tested": len(test_params),
                "parameters": test_params if not self.use_positional else None,
                "max_difference": float(np.max(np.abs(diff))),
                "conclusion": conclusion,
            },
        )

    def _get_params(self, result, param_names):
        """Extract parameter vector."""
        if isinstance(result.params, pd.Series):
            if self.use_positional:
                # Use positional indexing
                params = result.params.iloc[param_names].values
            else:
                # Use name indexing
                params = result.params[param_names].values
        else:
            # Numpy array
            params = np.asarray(result.params)
            if self.use_positional:
                params = params[param_names]

        return params

    def _get_covariance(self, result, param_names):
        """Extract covariance matrix for specified parameters."""
        # Try different ways to get covariance
        if hasattr(result, "cov_params"):
            if callable(result.cov_params):
                cov_full = result.cov_params()
            else:
                cov_full = result.cov_params
        elif hasattr(result, "cov"):
            cov_full = result.cov
        elif hasattr(result, "bse"):
            # Only standard errors available - construct diagonal covariance
            if isinstance(result.bse, pd.Series):
                if self.use_positional:
                    bse = result.bse.iloc[param_names].values
                else:
                    bse = result.bse[param_names].values
            else:
                bse = np.asarray(result.bse)
                if self.use_positional:
                    bse = bse[param_names]
            cov_full = np.diag(bse**2)
            return cov_full
        else:
            raise ValueError("Cannot extract covariance matrix from result")

        # Extract submatrix for parameters of interest
        if isinstance(cov_full, pd.DataFrame):
            if self.use_positional:
                cov = cov_full.iloc[param_names, param_names].values
            else:
                cov = cov_full.loc[param_names, param_names].values
        else:
            # Numpy array
            cov_full = np.asarray(cov_full)
            if self.use_positional:
                idx = param_names
            else:
                # Need to find indices
                if hasattr(result, "params") and isinstance(result.params, pd.Series):
                    all_params = list(result.params.index)
                    idx = [all_params.index(p) for p in param_names]
                else:
                    idx = param_names

            cov = cov_full[np.ix_(idx, idx)]

        return cov

    def _get_model_name(self, result):
        """Extract model name from result object."""
        if hasattr(result, "model"):
            if hasattr(result.model, "__class__"):
                return result.model.__class__.__name__
            elif hasattr(result.model, "name"):
                return result.model.name
        elif hasattr(result, "__class__"):
            return result.__class__.__name__
        return "Model"

    def summary(self) -> pd.DataFrame:
        """
        Create summary comparison table.

        Returns
        -------
        pd.DataFrame
            Comparison of parameter estimates
        """
        # Run test first if not done
        if not hasattr(self, "_last_result"):
            self._last_result = self.run()

        # Extract parameters for comparison
        params1 = self._get_params(self.result1, self.common_params)
        params2 = self._get_params(self.result2, self.common_params)

        # Get standard errors if available
        if hasattr(self.result1, "bse"):
            if isinstance(self.result1.bse, pd.Series):
                if self.use_positional:
                    se1 = self.result1.bse.iloc[self.common_params].values
                else:
                    se1 = self.result1.bse[self.common_params].values
            else:
                se1 = np.asarray(self.result1.bse)[self.common_params]
        else:
            se1 = np.nan * np.ones(len(params1))

        if hasattr(self.result2, "bse"):
            if isinstance(self.result2.bse, pd.Series):
                if self.use_positional:
                    se2 = self.result2.bse.iloc[self.common_params].values
                else:
                    se2 = self.result2.bse[self.common_params].values
            else:
                se2 = np.asarray(self.result2.bse)[self.common_params]
        else:
            se2 = np.nan * np.ones(len(params2))

        # Create DataFrame
        model1_name = self._get_model_name(self.result1)
        model2_name = self._get_model_name(self.result2)

        summary_df = pd.DataFrame(
            {
                "Parameter": self.common_params
                if not self.use_positional
                else [f"beta_{i}" for i in self.common_params],
                f"{model1_name}_coef": params1,
                f"{model1_name}_se": se1,
                f"{model2_name}_coef": params2,
                f"{model2_name}_se": se2,
                "Difference": params1 - params2,
                "Abs_Diff": np.abs(params1 - params2),
            }
        )

        return summary_df.round(6)
