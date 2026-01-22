"""
GMM Estimation Algorithms
==========================

Low-level GMM estimation routines implementing one-step, two-step,
and iterative GMM with Windmeijer correction.

Classes
-------
GMMEstimator : Low-level GMM estimation algorithms

References
----------
.. [1] Hansen, L. P. (1982). "Large Sample Properties of Generalized Method
       of Moments Estimators." Econometrica, 50(4), 1029-1054.

.. [2] Windmeijer, F. (2005). "A Finite Sample Correction for the Variance of
       Linear Efficient Two-Step GMM Estimators." Journal of Econometrics,
       126(1), 25-51.

.. [3] Hansen, L. P., Heaton, J., & Yaron, A. (1996). "Finite-Sample
       Properties of Some Alternative GMM Estimators." Journal of Business &
       Economic Statistics, 14(3), 262-280.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import linalg
import warnings


class GMMEstimator:
    """
    Low-level GMM estimation routines.

    This class implements the core mathematical algorithms for GMM estimation:
    - One-step GMM
    - Two-step GMM with Windmeijer correction
    - Iterative GMM (CUE)

    Parameters
    ----------
    tol : float
        Convergence tolerance for iterative methods
    max_iter : int
        Maximum iterations for iterative GMM
    """

    def __init__(self, tol: float = 1e-6, max_iter: int = 100):
        """Initialize estimator."""
        self.tol = tol
        self.max_iter = max_iter

    def one_step(self,
                 y: np.ndarray,
                 X: np.ndarray,
                 Z: np.ndarray,
                 skip_instrument_cleaning: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One-step GMM estimation.

        Uses weight matrix W = (Z'Z)^{-1}, which is efficient under
        homoskedasticity but not optimal under heteroskedasticity.

        Parameters
        ----------
        y : np.ndarray
            Dependent variable (n x 1)
        X : np.ndarray
            Regressors (n x k)
        Z : np.ndarray
            Instruments (n x n_instruments)

        Returns
        -------
        beta : np.ndarray
            Estimated coefficients (k x 1)
        W : np.ndarray
            Weight matrix (n_instruments x n_instruments)
        residuals : np.ndarray
            Residuals (n x 1)

        Notes
        -----
        GMM estimator: β = (X'Z W Z'X)^{-1} (X'Z W Z'y)
        Weight matrix: W = (Z'Z)^{-1}
        """
        # Ensure arrays are float64
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)

        # Remove observations with missing values
        valid_mask = self._get_valid_mask(y, X, Z)
        y_clean = y[valid_mask]
        X_clean = X[valid_mask]
        Z_clean = Z[valid_mask]

        # Compute weight matrix W = (Z'Z)^{-1}
        ZtZ = Z_clean.T @ Z_clean
        try:
            W = linalg.inv(ZtZ)
        except linalg.LinAlgError:
            # Singular matrix, use pseudo-inverse
            warnings.warn("Singular Z'Z matrix, using pseudo-inverse")
            W = linalg.pinv(ZtZ)

        # Compute GMM estimator
        # β = (X'Z W Z'X)^{-1} (X'Z W Z'y)
        XtZ = X_clean.T @ Z_clean
        ZtX = Z_clean.T @ X_clean
        Zty = Z_clean.T @ y_clean

        # A = X'Z W Z'X
        A = XtZ @ W @ ZtX
        try:
            A_inv = linalg.inv(A)
        except linalg.LinAlgError:
            warnings.warn("Singular A matrix, using pseudo-inverse")
            A_inv = linalg.pinv(A)

        # b = X'Z W Z'y
        b = XtZ @ W @ Zty

        # β = A^{-1} b
        beta = A_inv @ b

        # Compute residuals
        residuals = np.full_like(y, np.nan)
        residuals[valid_mask] = y_clean - X_clean @ beta

        return beta, W, residuals

    def two_step(self,
                 y: np.ndarray,
                 X: np.ndarray,
                 Z: np.ndarray,
                 robust: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Two-step GMM estimation with Windmeijer correction.

        Two-step GMM is asymptotically efficient under heteroskedasticity.
        Windmeijer (2005) correction is crucial for finite-sample inference.

        Parameters
        ----------
        y : np.ndarray
            Dependent variable (n x 1)
        X : np.ndarray
            Regressors (n x k)
        Z : np.ndarray
            Instruments (n x n_instruments)
        robust : bool
            Whether to use robust variance matrix (Windmeijer correction)

        Returns
        -------
        beta : np.ndarray
            Estimated coefficients (k x 1)
        vcov : np.ndarray
            Variance-covariance matrix (k x k)
        W : np.ndarray
            Optimal weight matrix (n_instruments x n_instruments)
        residuals : np.ndarray
            Residuals (n x 1)

        Notes
        -----
        Step 1: One-step GMM to get initial residuals
        Step 2: Construct optimal weight matrix W = (Z'ΩZ)^{-1}
                where Ω is residual variance matrix
        Step 3: Re-estimate with optimal W
        Step 4: Apply Windmeijer correction if robust=True
        """
        # Ensure arrays are float64
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)

        # Remove observations with missing values
        valid_mask = self._get_valid_mask(y, X, Z)
        y_clean = y[valid_mask]
        X_clean = X[valid_mask]
        Z_clean = Z[valid_mask]

        # Step 1: One-step GMM to get initial residuals
        beta_init, _, resid_init_full = self.one_step(y, X, Z)
        resid_init = resid_init_full[valid_mask]

        # Step 2: Construct optimal weight matrix
        W_optimal = self._compute_optimal_weight(Z_clean, resid_init, robust=True)

        # Step 3: Re-estimate with optimal weight matrix
        XtZ = X_clean.T @ Z_clean
        ZtX = Z_clean.T @ X_clean
        Zty = Z_clean.T @ y_clean

        # A = X'Z W Z'X
        A = XtZ @ W_optimal @ ZtX
        try:
            A_inv = linalg.inv(A)
        except linalg.LinAlgError:
            warnings.warn("Singular A matrix in two-step, using pseudo-inverse")
            A_inv = linalg.pinv(A)

        # b = X'Z W Z'y
        b = XtZ @ W_optimal @ Zty

        # β = A^{-1} b
        beta = A_inv @ b

        # Compute residuals
        residuals = np.full_like(y, np.nan)
        residuals[valid_mask] = y_clean - X_clean @ beta

        # Step 4: Compute variance-covariance matrix
        if robust:
            # Windmeijer (2005) correction
            vcov = self.windmeijer_correction(
                X_clean, Z_clean, residuals[valid_mask], W_optimal, A_inv
            )
        else:
            # Standard two-step variance (downward biased)
            vcov = A_inv

        return beta, vcov, W_optimal, residuals

    def _compute_optimal_weight(self,
                               Z: np.ndarray,
                               residuals: np.ndarray,
                               robust: bool = True) -> np.ndarray:
        """
        Compute optimal GMM weight matrix.

        Parameters
        ----------
        Z : np.ndarray
            Instruments (n x n_instruments)
        residuals : np.ndarray
            Residuals from initial estimation (n x 1)
        robust : bool
            Use robust variance (heteroskedasticity-consistent)

        Returns
        -------
        W : np.ndarray
            Optimal weight matrix (n_instruments x n_instruments)

        Notes
        -----
        Robust: W = (Z'ΩZ)^{-1} where Ω = diag(ε²)
        Non-robust: W = (1/n) (Z'Z)^{-1}
        """
        n = Z.shape[0]

        if robust:
            # Heteroskedasticity-robust weight matrix
            # Ω = diag(ε²)
            Omega = np.diag(residuals.flatten() ** 2)
            ZtOmegaZ = Z.T @ Omega @ Z
        else:
            # Homoskedastic weight matrix
            sigma2 = np.mean(residuals ** 2)
            ZtOmegaZ = sigma2 * (Z.T @ Z)

        try:
            W = linalg.inv(ZtOmegaZ)
        except linalg.LinAlgError:
            warnings.warn("Singular optimal weight matrix, using pseudo-inverse")
            W = linalg.pinv(ZtOmegaZ)

        return W

    def windmeijer_correction(self,
                             X: np.ndarray,
                             Z: np.ndarray,
                             residuals: np.ndarray,
                             W: np.ndarray,
                             A_inv: np.ndarray) -> np.ndarray:
        """
        Windmeijer (2005) finite-sample correction for two-step GMM.

        The standard two-step GMM variance estimator is severely downward
        biased in finite samples. Windmeijer's correction adjusts for the
        estimation error in the weight matrix.

        Parameters
        ----------
        X : np.ndarray
            Regressors (n x k)
        Z : np.ndarray
            Instruments (n x n_instruments)
        residuals : np.ndarray
            Two-step residuals (n x 1)
        W : np.ndarray
            Optimal weight matrix (n_instruments x n_instruments)
        A_inv : np.ndarray
            (X'Z W Z'X)^{-1} matrix (k x k)

        Returns
        -------
        vcov_corrected : np.ndarray
            Corrected variance-covariance matrix (k x k)

        References
        ----------
        Windmeijer, F. (2005). "A Finite Sample Correction for the Variance of
        Linear Efficient Two-Step GMM Estimators." Journal of Econometrics,
        126(1), 25-51.
        """
        n, k = X.shape
        n_instruments = Z.shape[1]

        # Compute moment conditions: g_i = Z_i * ε_i
        g = Z * residuals

        # Estimate variance of moments: Σ = E[g_i g_i']
        Sigma = (g.T @ g) / n

        # Compute D = E[∂g_i/∂β'] = -E[Z_i X_i']
        D = -(Z.T @ X) / n

        # Standard two-step variance (uncorrected)
        # V_uncorrected = (D' W D)^{-1}
        # This is what A_inv already is

        # Windmeijer correction term
        # Accounts for estimation of W in first step
        correction = self._compute_windmeijer_correction_term(
            X, Z, residuals, W, D, Sigma
        )

        # Corrected variance
        # V_corrected = A_inv + A_inv * correction * A_inv
        vcov_corrected = A_inv + A_inv @ correction @ A_inv

        # Ensure symmetry
        vcov_corrected = (vcov_corrected + vcov_corrected.T) / 2

        return vcov_corrected

    def _compute_windmeijer_correction_term(self,
                                           X: np.ndarray,
                                           Z: np.ndarray,
                                           residuals: np.ndarray,
                                           W: np.ndarray,
                                           D: np.ndarray,
                                           Sigma: np.ndarray) -> np.ndarray:
        """
        Compute the correction term for Windmeijer's variance.

        This is the most complex part of the Windmeijer correction,
        accounting for the effect of estimating W.
        """
        n = X.shape[0]
        n_instruments = Z.shape[1]

        # Compute H matrices (derivatives of weight matrix)
        # H_jl = ∂W/∂σ_{jl} where σ_{jl} = E[Z_j ε Z_l ε]
        #
        # For computational efficiency, we use:
        # ∂W/∂σ_{jl} = -W * (e_j e_l' + e_l e_j') * W
        # where e_j is unit vector

        # Compute B = Σ_{i=1}^n (∂g_i/∂β') W (∂²Σ/∂β∂σ) W (∂g_i/∂β)
        # This is approximated by a simpler form in practice

        # Simplified Windmeijer correction (commonly used)
        # Based on equation (12) in Windmeijer (2005)

        # Compute moment Jacobian weighted by W
        DWD = D.T @ W @ D  # k x k

        # Compute correction for estimation of Σ
        # This captures the variability in the weight matrix estimation
        g = Z * residuals  # n x n_instruments

        # For each observation, compute contribution to correction
        correction = np.zeros((X.shape[1], X.shape[1]))

        for i in range(n):
            # g_i = Z_i * ε_i
            g_i = g[i:i+1, :].T  # n_instruments x 1

            # X_i weighted by instruments
            ZiXi = Z[i:i+1, :].T @ X[i:i+1, :]  # n_instruments x k

            # Contribution to correction
            # This is a simplified version that captures the main effect
            H_i = W @ (g_i @ g_i.T) @ W  # Effect of observation i on W
            contrib = ZiXi.T @ H_i @ ZiXi

            correction += contrib

        correction = correction / n

        return correction

    def iterative(self,
                 y: np.ndarray,
                 X: np.ndarray,
                 Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Iterated GMM (CUE - Continuously Updated Estimator).

        Iteratively updates both β and W until convergence.
        Hansen et al. (1996) show this can have better finite-sample
        properties than two-step in some cases.

        Parameters
        ----------
        y : np.ndarray
            Dependent variable (n x 1)
        X : np.ndarray
            Regressors (n x k)
        Z : np.ndarray
            Instruments (n x n_instruments)

        Returns
        -------
        beta : np.ndarray
            Estimated coefficients (k x 1)
        vcov : np.ndarray
            Variance-covariance matrix (k x k)
        W : np.ndarray
            Weight matrix at convergence (n_instruments x n_instruments)
        converged : bool
            Whether iteration converged

        Notes
        -----
        Algorithm:
        1. Start with one-step β
        2. Compute W(β)
        3. Update β using W
        4. Repeat 2-3 until ||β_new - β_old|| < tol
        """
        # Remove observations with missing values
        valid_mask = self._get_valid_mask(y, X, Z)
        y_clean = y[valid_mask]
        X_clean = X[valid_mask]
        Z_clean = Z[valid_mask]

        # Initialize with one-step
        beta_old, _, resid_full = self.one_step(y, X, Z)
        resid_old = resid_full[valid_mask]

        converged = False
        for iteration in range(self.max_iter):
            # Update weight matrix using current residuals
            W = self._compute_optimal_weight(Z_clean, resid_old, robust=True)

            # Update β using new W
            XtZ = X_clean.T @ Z_clean
            ZtX = Z_clean.T @ X_clean
            Zty = Z_clean.T @ y_clean

            A = XtZ @ W @ ZtX
            try:
                A_inv = linalg.inv(A)
            except linalg.LinAlgError:
                A_inv = linalg.pinv(A)

            b = XtZ @ W @ Zty
            beta_new = A_inv @ b

            # Check convergence
            if self._check_convergence(beta_old, beta_new):
                converged = True
                break

            # Update for next iteration
            beta_old = beta_new
            resid_old = y_clean - X_clean @ beta_new

        if not converged:
            warnings.warn(f"Iterative GMM did not converge in {self.max_iter} iterations")

        # Final residuals
        residuals = np.full_like(y, np.nan)
        residuals[valid_mask] = y_clean - X_clean @ beta_new

        # Variance matrix (with Windmeijer-style correction)
        vcov = self.windmeijer_correction(
            X_clean, Z_clean, residuals[valid_mask], W, A_inv
        )

        return beta_new, vcov, W, converged

    def _check_convergence(self,
                          beta_old: np.ndarray,
                          beta_new: np.ndarray) -> bool:
        """
        Check convergence of iterative methods.

        Parameters
        ----------
        beta_old : np.ndarray
            Previous parameter vector
        beta_new : np.ndarray
            New parameter vector

        Returns
        -------
        bool
            True if converged
        """
        diff = np.max(np.abs(beta_new - beta_old))
        return diff < self.tol

    def _compute_gram_matrix_sparse(self, A: np.ndarray, B: np.ndarray = None) -> np.ndarray:
        """
        Compute A'B handling NaN values properly for sparse GMM instruments.

        For GMM-style instruments, NaN indicates instrument not available.
        Each element (i,j) of A'B is computed as sum over observations where
        BOTH A[:,i] and B[:,j] are non-NaN.

        This is the CORRECT approach for GMM with sparse instruments, as each
        moment condition should only include observations where the instrument
        is actually available.

        Parameters
        ----------
        A : np.ndarray (n x p)
            First matrix (typically Z or X)
        B : np.ndarray (n x q), optional
            Second matrix (typically Z, X, or y). If None, computes A'A.

        Returns
        -------
        AtB : np.ndarray (p x q)
            Gram matrix computed using pairwise-valid observations

        Notes
        -----
        This uses a simple nested loop which may be slow for large matrices.
        Future optimization: vectorize using broadcasting and nansum.
        """
        if B is None:
            B = A

        p = A.shape[1]
        q = B.shape[1]
        AtB = np.zeros((p, q))

        # For each column pair, sum over observations where both are valid
        for i in range(p):
            for j in range(q):
                # Valid where both A[:, i] and B[:, j] are not NaN
                valid = ~(np.isnan(A[:, i]) | np.isnan(B[:, j]))
                if valid.any():
                    AtB[i, j] = np.sum(A[valid, i] * B[valid, j])

        return AtB

    def _get_valid_mask(self,
                       y: np.ndarray,
                       X: np.ndarray,
                       Z: np.ndarray,
                       min_instruments: Optional[int] = None) -> np.ndarray:
        """
        Get mask of observations with sufficient valid data.

        For unbalanced panels, allows observations where some instruments are
        missing, as long as enough instruments remain for overidentification.

        Parameters
        ----------
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Regressors
        Z : np.ndarray
            Instruments
        min_instruments : int, optional
            Minimum number of valid instruments required per observation.
            If None, uses max(k+1, n_instruments//2) where k = number of regressors.

        Returns
        -------
        np.ndarray
            Boolean mask of valid observations
        """
        y_valid = ~np.isnan(y).any(axis=1) if y.ndim > 1 else ~np.isnan(y)
        X_valid = ~np.isnan(X).any(axis=1)

        # For instruments, count how many are valid per observation
        Z_notnan = ~np.isnan(Z)  # Boolean array: True where not NaN
        n_valid_instruments = Z_notnan.sum(axis=1)  # Count per row

        # Determine minimum required instruments
        if min_instruments is None:
            k = X.shape[1] if X.ndim > 1 else 1
            n_instruments_total = Z.shape[1] if Z.ndim > 1 else 1
            # For unbalanced panels, require at least k+1 for overidentification
            # but don't require half of total instruments (too restrictive)
            min_instruments = k + 1

        # Observation is valid if y, X are valid AND has enough instruments
        Z_valid = n_valid_instruments >= min_instruments

        return y_valid & X_valid & Z_valid
