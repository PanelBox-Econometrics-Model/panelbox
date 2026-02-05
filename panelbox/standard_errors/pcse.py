"""
Panel-Corrected Standard Errors (PCSE).

PCSE (Beck & Katz 1995) are designed for panel data with cross-sectional
dependence. They estimate the full cross-sectional covariance matrix of
the errors and use FGLS to obtain efficient standard errors.

PCSE requires T > N (more time periods than entities).
"""

from typing import Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .utils import compute_bread


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
        self,
        X: np.ndarray,
        resid: np.ndarray,
        entity_ids: np.ndarray,
        time_ids: np.ndarray
    ):
        self.X = X
        self.resid = resid
        self.entity_ids = np.asarray(entity_ids)
        self.time_ids = np.asarray(time_ids)

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.entity_ids) != self.n_obs:
            raise ValueError(
                f"entity_ids dimension mismatch: expected {self.n_obs}, "
                f"got {len(self.entity_ids)}"
            )
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, "
                f"got {len(self.time_ids)}"
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
                UserWarning
            )

    def _reshape_panel(self) -> np.ndarray:
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

    def _estimate_sigma(self) -> np.ndarray:
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

    def compute(self) -> PCSEResult:
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
                "Σ matrix is singular. Using pseudoinverse. "
                "Results may be unreliable.",
                UserWarning
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
                UserWarning
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
            n_periods=self.n_periods
        )

    def diagnostic_summary(self) -> str:
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
            lines.append(f"  PCSE requires T > N")
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


def pcse(
    X: np.ndarray,
    resid: np.ndarray,
    entity_ids: np.ndarray,
    time_ids: np.ndarray
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
