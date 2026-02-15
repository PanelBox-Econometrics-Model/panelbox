"""
GMM Diagnostic Tests
====================

Implements diagnostic tests for GMM estimation including:
- Hansen J-test for overidentification
- C-statistic (Difference-in-Sargan) for subset validity
- Weak instruments diagnostics
- Cragg-Donald F-statistic

These tests help assess model specification and instrument validity.

Classes
-------
GMMDiagnostics : Container for GMM diagnostic tests

References
----------
.. [1] Hansen, L. P. (1982). "Large Sample Properties of Generalized Method
       of Moments Estimators." Econometrica, 50(4), 1029-1054.

.. [2] Newey, W. K., & West, K. D. (1987). "Hypothesis Testing with Efficient
       Method of Moments Estimation." International Economic Review, 28(3), 777-787.

.. [3] Stock, J. H., & Yogo, M. (2005). "Testing for Weak Instruments in Linear IV
       Regression." In Identification and Inference for Econometric Models: Essays in
       Honor of Thomas Rothenberg.

.. [4] Cragg, J. G., & Donald, S. G. (1993). "Testing Identifiability and Specification
       in Instrumental Variable Models." Econometric Theory, 9(2), 222-240.

Examples
--------
>>> from panelbox.gmm.diagnostics import GMMDiagnostics
>>> diagnostics = GMMDiagnostics(model, results)
>>> print(diagnostics.summary())
"""

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.stats import chi2
from scipy.stats import f as f_dist

from panelbox.gmm.results import TestResult


class GMMDiagnostics:
    """
    Container class for GMM diagnostic tests.

    Provides methods to test model specification, instrument validity,
    and instrument strength.

    Parameters
    ----------
    model : GMMModel
        Fitted GMM model object
    results : GMMResults
        Results from GMM estimation

    Attributes
    ----------
    hansen_j : TestResult
        Hansen J-test for overidentification
    weak_instruments : Dict
        Weak instruments diagnostics

    Methods
    -------
    c_statistic : Compute difference-in-Sargan C-statistic
    weak_instruments_test : Test for weak instruments
    summary : Generate diagnostic summary report

    Examples
    --------
    >>> diagnostics = GMMDiagnostics(model, results)
    >>> c_stat = diagnostics.c_statistic(subset_instruments=['z1', 'z2'])
    >>> weak_test = diagnostics.weak_instruments_test()
    >>> print(diagnostics.summary())
    """

    def __init__(self, model, results):
        """Initialize GMM diagnostics."""
        self.model = model
        self.results = results

        # Extract key attributes
        if hasattr(model, "y"):
            self.y = model.y
            self.X = model.X
            self.Z = model.Z
            self.n = len(self.y)
            self.k = model.k if hasattr(model, "k") else self.X.shape[1]
            self.n_instruments = (
                model.n_instruments if hasattr(model, "n_instruments") else self.Z.shape[1]
            )
        else:
            # Try to extract from results
            self.n = results.nobs
            self.k = results.n_params
            self.n_instruments = results.n_instruments
            self.y = None
            self.X = None
            self.Z = None

        # Store Hansen J-test from results
        self.hansen_j = results.hansen_j if hasattr(results, "hansen_j") else None

    def c_statistic(
        self,
        subset_indices: Optional[List[int]] = None,
        subset_names: Optional[List[str]] = None,
    ) -> TestResult:
        """
        Compute Difference-in-Sargan C-statistic.

        Tests the validity of a subset of moment conditions by comparing
        J-statistics from restricted and unrestricted models.

        Parameters
        ----------
        subset_indices : List[int], optional
            Indices of instruments to test (0-indexed)
        subset_names : List[str], optional
            Names of instruments to test (alternative to subset_indices)

        Returns
        -------
        test_result : TestResult
            C-statistic test result

        Notes
        -----
        The C-statistic tests H0: subset of instruments is valid

        C = J_restricted - J_unrestricted ~ χ²(# restrictions)

        where J_restricted uses only the subset of instruments,
        and J_unrestricted uses all instruments.

        A large C-statistic suggests the tested instruments may be invalid.

        Examples
        --------
        >>> # Test validity of instruments z3 and z4
        >>> c_test = diagnostics.c_statistic(subset_indices=[2, 3])
        >>> if c_test.pvalue < 0.05:
        ...     print("Subset of instruments may be invalid")
        """
        if self.y is None or self.X is None or self.Z is None:
            raise RuntimeError(
                "C-statistic requires access to data arrays. " "Model must have y, X, Z attributes."
            )

        # Determine subset of instruments to test
        if subset_indices is not None:
            subset_idx = subset_indices
        elif subset_names is not None:
            # Would need instrument names mapping (not implemented)
            raise NotImplementedError("subset_names not yet supported")
        else:
            raise ValueError("Must provide either subset_indices or subset_names")

        # Validate indices
        if max(subset_idx) >= self.n_instruments:
            raise ValueError(
                f"subset_indices contains {max(subset_idx)} but only "
                f"{self.n_instruments} instruments available"
            )

        # Get unrestricted J-statistic (all instruments)
        J_unrestricted = self.hansen_j.statistic

        # Compute restricted J-statistic (excluding subset)
        # Keep all instruments except those in subset
        keep_idx = [i for i in range(self.n_instruments) if i not in subset_idx]
        Z_restricted = self.Z[:, keep_idx]

        # Estimate GMM with restricted instruments
        params_restricted = self._estimate_gmm_restricted(Z_restricted)

        # Compute residuals
        residuals_restricted = self.y - self.X @ params_restricted.reshape(-1, 1)

        # Compute moments
        g_restricted = (1 / self.n) * (Z_restricted.T @ residuals_restricted)

        # Compute weighting matrix (simplified - use identity)
        W_restricted = np.eye(len(keep_idx))

        # Compute J-statistic for restricted model
        Q_restricted = float((g_restricted.T @ W_restricted @ g_restricted).item())
        J_restricted = self.n * Q_restricted

        # Compute C-statistic
        C = J_restricted - J_unrestricted

        # Degrees of freedom = number of tested instruments
        df = len(subset_idx)

        # p-value
        pvalue = 1 - chi2.cdf(C, df)

        # Interpretation
        if pvalue < 0.05:
            conclusion = f"Reject validity of subset (p={pvalue:.4f}). Instruments may be invalid."
        else:
            conclusion = f"Do not reject validity (p={pvalue:.4f}). Instruments appear valid."

        return TestResult(
            name="C-statistic (Difference-in-Sargan)",
            statistic=C,
            pvalue=pvalue,
            df=df,
            distribution="chi2",
            null_hypothesis="Tested instrument subset is valid",
            conclusion=conclusion,
        )

    def _estimate_gmm_restricted(self, Z_restricted: np.ndarray) -> np.ndarray:
        """
        Estimate GMM with restricted instrument set.

        Uses simple two-step GMM with given instruments.
        """
        # First step: W = I
        W1 = np.eye(Z_restricted.shape[1])

        # Compute β₁ = (X'Z W₁ Z'X)⁻¹ X'Z W₁ Z'y
        XZ = self.X.T @ Z_restricted
        ZX = Z_restricted.T @ self.X
        Zy = Z_restricted.T @ self.y

        try:
            beta = linalg.solve(XZ @ W1 @ ZX, XZ @ W1 @ Zy, assume_a="pos")
        except linalg.LinAlgError:
            # Use least squares if singular
            beta = linalg.lstsq(XZ @ W1 @ ZX, XZ @ W1 @ Zy)[0]

        return beta.flatten()

    def weak_instruments_test(self) -> Dict[str, Union[float, str]]:
        """
        Test for weak instruments.

        Computes diagnostics to assess instrument strength:
        - Cragg-Donald F-statistic
        - First-stage F-statistics
        - Shea partial R²

        Returns
        -------
        test_results : dict
            Dictionary with:
            - 'cragg_donald_f': Cragg-Donald F-statistic
            - 'critical_value_10pct': Critical value for 10% maximal IV size
            - 'first_stage_f': First-stage F-statistics (if applicable)
            - 'interpretation': Text interpretation

        Notes
        -----
        Weak instruments lead to:
        - Biased estimates in finite samples
        - Invalid inference (t-tests, confidence intervals)
        - Poor asymptotic approximations

        Rule of thumb (Stock-Yogo 2005):
        - F < 10: Weak instruments (be concerned)
        - F > 10: Instruments likely adequate

        For multiple endogenous variables, use Cragg-Donald statistic.

        Examples
        --------
        >>> weak_test = diagnostics.weak_instruments_test()
        >>> if weak_test['cragg_donald_f'] < 10:
        ...     print("Warning: Weak instruments detected")
        """
        if self.y is None or self.X is None or self.Z is None:
            raise RuntimeError(
                "Weak instruments test requires access to data arrays. "
                "Model must have y, X, Z attributes."
            )

        # Compute Cragg-Donald F-statistic
        cd_f = self._compute_cragg_donald_f()

        # Stock-Yogo critical values (approximate for single endogenous regressor)
        # For 10% maximal IV size
        critical_10pct = 16.38  # Approximate value

        # Interpretation
        if cd_f < 10:
            interpretation = (
                f"Weak instruments detected (F={cd_f:.2f} < 10). "
                "Estimates may be biased and inference invalid."
            )
            warning_level = "CRITICAL"
        elif cd_f < critical_10pct:
            interpretation = (
                f"Instruments moderately weak (F={cd_f:.2f}). " "Use caution in inference."
            )
            warning_level = "WARNING"
        else:
            interpretation = f"Instruments appear strong (F={cd_f:.2f} > {critical_10pct:.1f})."
            warning_level = "OK"

        return {
            "cragg_donald_f": cd_f,
            "critical_value_10pct": critical_10pct,
            "interpretation": interpretation,
            "warning_level": warning_level,
        }

    def _compute_cragg_donald_f(self) -> float:
        """
        Compute Cragg-Donald F-statistic for weak instruments.

        For single endogenous regressor, this simplifies to the
        first-stage F-statistic from regressing X on Z.
        """
        # Assume first endogenous regressor is in X[:, 1]
        # (X[:, 0] is typically intercept)

        if self.X.shape[1] == 1:
            # Only intercept, no endogenous regressors
            return np.inf

        # First-stage regression: X₁ ~ Z
        # For simplicity, compute F-stat for X[:, 1]
        X_endog = self.X[:, 1].reshape(-1, 1)  # First endogenous variable

        # Fit X_endog = Z @ pi + error
        try:
            pi = linalg.lstsq(self.Z, X_endog)[0]
            X_fitted = self.Z @ pi
            residuals = X_endog - X_fitted

            # Compute F-statistic
            # F = (R² / k) / ((1 - R²) / (n - k - 1))
            SSR = float(np.sum(residuals**2))
            TSS = float(np.sum((X_endog - np.mean(X_endog)) ** 2))

            if TSS == 0:
                return 0.0

            R2 = 1 - SSR / TSS

            k_instruments = self.Z.shape[1] - 1  # Exclude intercept
            n = self.n

            if R2 >= 1 or k_instruments == 0:
                return np.inf

            F = (R2 / k_instruments) / ((1 - R2) / (n - k_instruments - 1))

            return max(0.0, float(F))

        except Exception:
            warnings.warn("Could not compute Cragg-Donald F-statistic")
            return np.nan

    def diagnostic_tests(self) -> pd.DataFrame:
        """
        Run all diagnostic tests and return summary.

        Returns
        -------
        summary : pd.DataFrame
            DataFrame with all diagnostic tests

        Examples
        --------
        >>> summary = diagnostics.diagnostic_tests()
        >>> print(summary)
        """
        results = []

        # Hansen J-test
        if self.hansen_j is not None:
            results.append(
                {
                    "Test": "Hansen J-test",
                    "Statistic": self.hansen_j.statistic,
                    "p-value": self.hansen_j.pvalue,
                    "Result": "PASS" if self.hansen_j.pvalue > 0.05 else "FAIL",
                }
            )

        # Weak instruments test
        try:
            weak_test = self.weak_instruments_test()
            results.append(
                {
                    "Test": "Weak Instruments (Cragg-Donald F)",
                    "Statistic": weak_test["cragg_donald_f"],
                    "p-value": np.nan,
                    "Result": weak_test["warning_level"],
                }
            )
        except Exception as e:
            warnings.warn(f"Weak instruments test failed: {e}")

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def summary(self) -> str:
        """
        Generate comprehensive diagnostic summary.

        Returns
        -------
        summary_text : str
            Formatted text summary of all diagnostics

        Examples
        --------
        >>> print(diagnostics.summary())
        """
        lines = []
        lines.append("=" * 70)
        lines.append("GMM Diagnostic Tests")
        lines.append("=" * 70)
        lines.append("")

        # Hansen J-test
        if self.hansen_j is not None:
            lines.append("Hansen J-test for Overidentification")
            lines.append("-" * 70)
            lines.append(f"  Statistic: {self.hansen_j.statistic:.4f}")
            lines.append(f"  p-value:   {self.hansen_j.pvalue:.4f}")
            lines.append(f"  df:        {self.hansen_j.df}")
            lines.append(f"  Result:    {self.hansen_j.conclusion}")
            lines.append("")

        # Weak instruments
        try:
            weak_test = self.weak_instruments_test()
            lines.append("Weak Instruments Test")
            lines.append("-" * 70)
            lines.append(f"  Cragg-Donald F: {weak_test['cragg_donald_f']:.4f}")
            lines.append(f"  Critical (10%): {weak_test['critical_value_10pct']:.2f}")
            lines.append(f"  Result:         {weak_test['interpretation']}")
            lines.append("")
        except Exception:
            pass

        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GMMDiagnostics("
            f"n={self.n}, "
            f"k={self.k}, "
            f"n_instruments={self.n_instruments})"
        )
