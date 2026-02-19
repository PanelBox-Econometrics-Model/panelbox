"""
GMM Estimation Algorithms
==========================

Low-level GMM estimation routines implementing one-step, two-step,
and iterative GMM with per-individual weight matrices and
Windmeijer (2005) finite-sample correction.

The key corrections vs the original implementation:
- First-step weight uses H tridiagonal matrix: W₁ = Σᵢ Zᵢ' Hᵢ Zᵢ
- Second-step weight is clustered by individual: W₂ = (1/N) Σᵢ (Zᵢ'ûᵢ)(Zᵢ'ûᵢ)'
- Windmeijer correction follows pydynpd (validated against Stata xtabond2)

References
----------
.. [1] Hansen, L. P. (1982). "Large Sample Properties of Generalized Method
       of Moments Estimators." Econometrica, 50(4), 1029-1054.

.. [2] Windmeijer, F. (2005). "A Finite Sample Correction for the Variance of
       Linear Efficient Two-Step GMM Estimators." Journal of Econometrics,
       126(1), 25-51.

.. [3] Arellano, M., & Bond, S. (1991). "Some Tests of Specification for Panel
       Data: Monte Carlo Evidence and an Application to Employment Equations."
       Review of Economic Studies, 58(2), 277-297.
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import linalg


class GMMEstimator:
    """
    Low-level GMM estimation routines for panel data.

    Implements one-step, two-step, and iterative GMM following the
    Arellano-Bond/Blundell-Bond framework with:
    - Per-individual weight matrix computation
    - H matrix for first-step (first-difference transformation)
    - Clustered second-step weight matrix
    - Windmeijer (2005) finite-sample correction

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

        # Results stored after estimation (used by Hansen J test)
        self.N: Optional[int] = None
        self.zs: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.W2_inv: Optional[np.ndarray] = None
        self._step1_M_XZ_W: Optional[np.ndarray] = None

    @staticmethod
    def build_H_matrix(T: int, transformation: str = "fd") -> np.ndarray:
        """
        Build the H matrix for first-step GMM weight.

        For first-differences, H captures E[Δu Δu']/σ² under iid errors:
        H = [2, -1, 0, ...; -1, 2, -1, ...; 0, -1, 2, ...]

        Parameters
        ----------
        T : int
            Number of time periods (after differencing)
        transformation : str
            'fd' for first-difference, 'fod' for forward orthogonal deviations

        Returns
        -------
        H : np.ndarray (T × T)
        """
        if transformation == "fd":
            H = np.zeros((T, T))
            for i in range(T):
                H[i, i] = 2.0
                if i > 0:
                    H[i, i - 1] = -1.0
                if i < T - 1:
                    H[i, i + 1] = -1.0
            return H
        else:
            return np.eye(T)

    def one_step(
        self,
        y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        ids: Optional[np.ndarray] = None,
        H_blocks: Optional[Dict] = None,
        skip_instrument_cleaning: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One-step GMM estimation.

        With ids: W₁ = Σᵢ Zᵢ' Hᵢ Zᵢ, β = (X'Z W₁⁻¹ Z'X)⁻¹ X'Z W₁⁻¹ Z'y
        Without ids (legacy): W = (Z'Z), β = (X'Z W⁻¹ Z'X)⁻¹ X'Z W⁻¹ Z'y

        Parameters
        ----------
        y : np.ndarray
            Dependent variable (n x 1)
        X : np.ndarray
            Regressors (n x k)
        Z : np.ndarray
            Instruments (n x L)
        ids : np.ndarray, optional
            Individual identifiers for per-individual weight computation.
        H_blocks : dict, optional
            Pre-computed H matrices keyed by individual id.
            If None and ids provided, builds tridiagonal H automatically.

        Returns
        -------
        beta : np.ndarray
            Estimated coefficients (k x 1)
        W_inv : np.ndarray
            Weight matrix used in estimation (L x L)
        residuals : np.ndarray
            Residuals (n x 1)
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # For legacy API (no ids): filter NaN rows and restore them in residuals
        valid_mask = None
        if ids is None:
            valid_mask = self._get_valid_mask(y, X, Z)
            if not valid_mask.all():
                y_orig = y
                y = y[valid_mask]
                X = X[valid_mask]
                Z = Z[valid_mask]

        if ids is not None:
            unique_ids = np.unique(ids)
            L = Z.shape[1]

            # Build H blocks if not provided
            if H_blocks is None:
                H_blocks = {}
                for uid in unique_ids:
                    T_i = int(np.sum(ids == uid))
                    H_blocks[uid] = self.build_H_matrix(T_i)

            # W₁ = Σᵢ Zᵢ' Hᵢ Zᵢ (no 1/N factor, following pydynpd)
            W1 = np.zeros((L, L))
            for uid in unique_ids:
                mask = ids == uid
                Z_i = Z[mask]
                H_i = H_blocks[uid]
                W1 += Z_i.T @ H_i @ Z_i

            W_inv = self._safe_pinv(W1)
        else:
            # Legacy: W = Z'Z
            ZtZ = Z.T @ Z
            W_inv = self._safe_pinv(ZtZ)

        # GMM estimator: β = (X'Z W⁻¹ Z'X)⁻¹ X'Z W⁻¹ Z'y
        XtZ = X.T @ Z  # (K × L)
        ZtX = Z.T @ X  # (L × K)
        Zty = Z.T @ y  # (L × 1)

        XZ_W = XtZ @ W_inv  # (K × L)
        M_inv = XZ_W @ ZtX  # (K × K) = X'Z W⁻¹ Z'X
        M = self._safe_pinv(M_inv)
        M_XZ_W = M @ XZ_W  # (K × L)

        beta = M_XZ_W @ Zty  # (K × 1)
        residuals_clean = y - X @ beta

        # For legacy API: restore NaN in residuals for filtered rows
        if valid_mask is not None and not valid_mask.all():
            residuals = np.full_like(y_orig, np.nan)
            residuals[valid_mask] = residuals_clean
        else:
            residuals = residuals_clean

        # Store step-1 projection for later use (robust vcov, Windmeijer)
        self._step1_M_XZ_W = M_XZ_W

        return beta, W_inv, residuals

    def compute_one_step_robust_vcov(
        self,
        Z: np.ndarray,
        residuals: np.ndarray,
        ids: np.ndarray,
    ) -> np.ndarray:
        """
        Compute robust (sandwich) vcov for one-step GMM.

        V₁ = N × M_XZ_W × W_next × M_XZ_W'
        where W_next = (1/N) Σᵢ (Zᵢ'ûᵢ)(Zᵢ'ûᵢ)'

        Must be called after one_step().

        Parameters
        ----------
        Z : np.ndarray (n, L)
        residuals : np.ndarray (n, 1)
        ids : np.ndarray (n,)

        Returns
        -------
        vcov : np.ndarray (k, k)
        """
        if self._step1_M_XZ_W is None:
            raise ValueError("Must call one_step() before compute_one_step_robust_vcov()")

        N = len(np.unique(ids))
        W_next, W_next_inv, zs = self._compute_clustered_weight(Z, residuals, ids)
        vcov = N * (self._step1_M_XZ_W @ W_next @ self._step1_M_XZ_W.T)
        vcov = (vcov + vcov.T) / 2

        # Store for Hansen J
        self.N = N
        self.W2 = W_next
        self.W2_inv = W_next_inv
        self.zs = zs

        return vcov

    def two_step(
        self,
        y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        ids: Optional[np.ndarray] = None,
        H_blocks: Optional[Dict] = None,
        robust: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Two-step GMM estimation with Windmeijer (2005) correction.

        Step 1: Estimate β₁ using H-based weight matrix
        Step 2: Compute optimal weight W₂ = (1/N) Σᵢ (Zᵢ'û₁ᵢ)(Zᵢ'û₁ᵢ)'
        Step 3: Re-estimate β₂ using W₂
        Step 4: Apply Windmeijer correction for variance

        Parameters
        ----------
        y : np.ndarray (n, 1)
        X : np.ndarray (n, k)
        Z : np.ndarray (n, L)
        ids : np.ndarray (n,), optional
        H_blocks : dict, optional
        robust : bool
            Whether to apply Windmeijer correction (default True)

        Returns
        -------
        beta : np.ndarray (k, 1)
        vcov : np.ndarray (k, k)
        W2_inv : np.ndarray (L, L)
        residuals : np.ndarray (n, 1)
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # For legacy API (no ids): filter NaN rows
        valid_mask = None
        if ids is None:
            valid_mask = self._get_valid_mask(y, X, Z)
            if not valid_mask.all():
                y_orig = y
                y = y[valid_mask]
                X = X[valid_mask]
                Z = Z[valid_mask]

        # Step 1: One-step GMM
        beta_1, W1_inv, resid_1_full = self.one_step(y, X, Z, ids, H_blocks)
        step1_M_XZ_W = self._step1_M_XZ_W.copy()
        # Get clean residuals (without NaN padding)
        resid_1 = y - X @ beta_1

        # Step 2: Compute optimal weight from step-1 residuals
        if ids is not None:
            unique_ids = np.unique(ids)
            N = len(unique_ids)
            W2, W2_inv, _ = self._compute_clustered_weight(Z, resid_1, ids)
        else:
            N = Z.shape[0]
            W2 = self._compute_weight_legacy(Z, resid_1)
            W2_inv = self._safe_pinv(W2)

        # Step 3: Re-estimate with optimal weight
        XtZ = X.T @ Z
        ZtX = Z.T @ X
        Zty = Z.T @ y

        XZ_W2 = XtZ @ W2_inv  # (K × L)
        M2_inv = XZ_W2 @ ZtX  # (K × K)
        M2 = self._safe_pinv(M2_inv)
        M2_XZ_W2 = M2 @ XZ_W2  # (K × L)

        beta_2 = M2_XZ_W2 @ Zty  # (K × 1)
        resid_2 = y - X @ beta_2

        # Step-2 per-individual moments (for Hansen J)
        if ids is not None:
            _, _, zs_2 = self._compute_clustered_weight(Z, resid_2, ids)
        else:
            zs_2 = Z.T @ resid_2

        # Store for Hansen J test
        self.N = N
        self.W2 = W2
        self.W2_inv = W2_inv
        self.zs = zs_2

        # Step 4: Variance-covariance
        if robust and ids is not None:
            # Step-1 robust vcov: V₁ = N × M_XZ_W₁ × W₂ × M_XZ_W₁'
            vcov_step1 = N * (step1_M_XZ_W @ W2 @ step1_M_XZ_W.T)
            vcov = self._windmeijer_correction(
                M2, M2_XZ_W2, W2_inv, zs_2, vcov_step1, X, Z, resid_1, ids
            )
        elif robust:
            # Legacy simplified Windmeijer (observation-level, no clustering)
            vcov = self._windmeijer_legacy(X, Z, resid_2, W2_inv, M2)
        else:
            vcov = M2

        # For legacy API: restore NaN in residuals for filtered rows
        if valid_mask is not None and not valid_mask.all():
            residuals = np.full_like(y_orig, np.nan)
            residuals[valid_mask] = resid_2
        else:
            residuals = resid_2

        return beta_2, vcov, W2_inv, residuals

    def iterative(
        self,
        y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        ids: Optional[np.ndarray] = None,
        H_blocks: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Iterated GMM (Continuously Updated Estimator).

        Parameters
        ----------
        y, X, Z : np.ndarray
        ids : np.ndarray, optional
        H_blocks : dict, optional

        Returns
        -------
        beta : np.ndarray (k, 1)
        vcov : np.ndarray (k, k)
        W_inv : np.ndarray (L, L)
        converged : bool
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # For legacy API (no ids): filter NaN rows
        valid_mask = None
        if ids is None:
            valid_mask = self._get_valid_mask(y, X, Z)
            if not valid_mask.all():
                y = y[valid_mask]
                X = X[valid_mask]
                Z = Z[valid_mask]

        # Initialize with one-step (pass ids so it uses per-individual weight)
        beta_old, _, resid_full = self.one_step(y, X, Z, ids, H_blocks)
        # For legacy mode, one_step returns full-length residuals with NaN;
        # we need the clean version for iterative updates
        resid_old = y - X @ beta_old

        XtZ = X.T @ Z
        ZtX = Z.T @ X
        Zty = Z.T @ y

        converged = False
        W_inv = None
        M = None

        for _iteration in range(self.max_iter):
            # Update weight from current residuals
            if ids is not None:
                _, W_inv, _ = self._compute_clustered_weight(Z, resid_old, ids)
            else:
                W = self._compute_weight_legacy(Z, resid_old)
                W_inv = self._safe_pinv(W)

            # Update β
            XZ_W = XtZ @ W_inv
            M_inv = XZ_W @ ZtX
            M = self._safe_pinv(M_inv)
            beta_new = M @ XZ_W @ Zty

            if self._check_convergence(beta_old, beta_new):
                converged = True
                break

            beta_old = beta_new
            resid_old = y - X @ beta_new

        if not converged:
            warnings.warn(f"Iterative GMM did not converge in {self.max_iter} iterations")

        residuals = y - X @ beta_new

        # Final weight and stored results for Hansen J
        if ids is not None:
            W_final, W_final_inv, zs = self._compute_clustered_weight(Z, residuals, ids)
            self.N = len(np.unique(ids))
            self.W2 = W_final
            self.W2_inv = W_final_inv
            self.zs = zs
        else:
            W_final_inv = W_inv

        vcov = M

        return beta_new, vcov, W_final_inv, converged

    # ------------------------------------------------------------------
    # Per-individual weight computation
    # ------------------------------------------------------------------

    def _compute_clustered_weight(
        self,
        Z: np.ndarray,
        residuals: np.ndarray,
        ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute clustered (per-individual) weight matrix.

        W₂ = (1/N) Σᵢ (Zᵢ'ûᵢ)(Zᵢ'ûᵢ)'
        zs = Σᵢ Zᵢ'ûᵢ

        Parameters
        ----------
        Z : np.ndarray (n, L)
        residuals : np.ndarray (n, 1) or (n,)
        ids : np.ndarray (n,)

        Returns
        -------
        W2 : np.ndarray (L, L)
            The "meat" matrix
        W2_inv : np.ndarray (L, L)
            Pseudo-inverse (the weight for estimation)
        zs : np.ndarray (L, 1)
            Sum of per-individual moment conditions
        """
        unique_ids = np.unique(ids)
        N = len(unique_ids)
        L = Z.shape[1]
        resid = residuals.reshape(-1, 1) if residuals.ndim == 1 else residuals

        W2 = np.zeros((L, L))
        zs = np.zeros((L, 1))

        for uid in unique_ids:
            mask = ids == uid
            Z_i = Z[mask]  # (T_i × L)
            u_i = resid[mask]  # (T_i × 1)
            m_i = Z_i.T @ u_i  # (L × 1)
            W2 += m_i @ m_i.T  # (L × L)
            zs += m_i

        W2 /= N
        W2_inv = self._safe_pinv(W2)

        return W2, W2_inv, zs

    # ------------------------------------------------------------------
    # Windmeijer (2005) correction
    # ------------------------------------------------------------------

    def _windmeijer_correction(
        self,
        M2: np.ndarray,
        M2_XZ_W2: np.ndarray,
        W2_inv: np.ndarray,
        zs2: np.ndarray,
        vcov_step1: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        resid_1: np.ndarray,
        ids: np.ndarray,
    ) -> np.ndarray:
        """
        Windmeijer (2005) finite-sample correction for two-step GMM.

        Follows pydynpd implementation (validated against Stata xtabond2).

        The correction accounts for the estimation error in the second-step
        weight matrix W₂, which causes the naive two-step variance to be
        severely downward biased.

        Parameters
        ----------
        M2 : (K × K) - (X'Z W₂⁻¹ Z'X)⁻¹
        M2_XZ_W2 : (K × L) - M₂ × X'Z × W₂⁻¹
        W2_inv : (L × L) - pseudo-inverse of W₂
        zs2 : (L × 1) - sum of step-2 per-individual moments
        vcov_step1 : (K × K) - step-1 robust vcov
        X : (n × K)
        Z : (n × L)
        resid_1 : (n × 1) - step-1 residuals
        ids : (n,) - individual identifiers

        Returns
        -------
        vcov : (K × K) - Windmeijer-corrected variance-covariance
        """
        unique_ids = np.unique(ids)
        N = len(unique_ids)
        K = X.shape[1]
        L = Z.shape[1]
        resid_1 = resid_1.reshape(-1, 1) if resid_1.ndim == 1 else resid_1

        D = np.empty((K, K), dtype=np.float64)

        for j in range(K):
            # Compute derivative of W₂ w.r.t. βⱼ
            zxz = np.zeros((L, L))

            for uid in unique_ids:
                mask = ids == uid
                X_i = X[mask]  # (T_i × K)
                u_1i = resid_1[mask]  # (T_i × 1)
                Z_i = Z[mask]  # (T_i × L)

                # xu = x_ij @ u_1i' = (T_i × 1) @ (1 × T_i) = (T_i × T_i)
                xu = X_i[:, j : j + 1] @ u_1i.T
                # Z_i' @ (xu + xu') @ Z_i = (L × L)
                zxz += Z_i.T @ (xu + xu.T) @ Z_i

            partial_dir = (-1.0 / N) * zxz

            # Dj = -M2_XZ_W2 @ partial_dir @ W2_inv @ zs2
            Dj = -1.0 * np.linalg.multi_dot([M2_XZ_W2, partial_dir, W2_inv, zs2])
            D[:, j : j + 1] = Dj

        # Corrected variance (exact pydynpd formula)
        D_M = D @ M2
        vcov = N * M2 + N * D_M + N * D_M.T + D @ vcov_step1 @ D.T

        # Ensure symmetry
        vcov = (vcov + vcov.T) / 2

        return vcov

    # ------------------------------------------------------------------
    # Legacy methods (backward compatibility when ids not provided)
    # ------------------------------------------------------------------

    def _compute_weight_legacy(self, Z: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        """Legacy observation-level weight matrix: Z' diag(e²) Z."""
        resid = residuals.flatten()
        Omega = np.diag(resid**2)
        return Z.T @ Omega @ Z

    def _windmeijer_legacy(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        residuals: np.ndarray,
        W: np.ndarray,
        A_inv: np.ndarray,
    ) -> np.ndarray:
        """Legacy simplified Windmeijer correction (observation-level)."""
        n = X.shape[0]
        resid = residuals.flatten() if residuals.ndim > 1 else residuals
        g = Z * resid.reshape(-1, 1)

        correction = np.zeros((X.shape[1], X.shape[1]))
        for i in range(n):
            g_i = g[i : i + 1, :].T
            ZiXi = Z[i : i + 1, :].T @ X[i : i + 1, :]
            H_i = W @ (g_i @ g_i.T) @ W
            correction += ZiXi.T @ H_i @ ZiXi
        correction /= n

        vcov = A_inv + A_inv @ correction @ A_inv
        return (vcov + vcov.T) / 2

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_pinv(A: np.ndarray) -> np.ndarray:
        """Invert matrix, using pseudo-inverse when ill-conditioned.

        When N (groups) < L (instruments), the weight matrix is rank-deficient.
        ``linalg.inv`` may succeed without raising an exception but return
        numerically garbage results.  We detect this via condition number
        and fall back to the Moore–Penrose pseudo-inverse (SVD-based).
        """
        try:
            result = linalg.inv(A)
            result = np.asarray(result)
            # Check if the inverse is numerically reliable
            cond = np.linalg.norm(A) * np.linalg.norm(result)
            if cond > 1e12:
                warnings.warn(
                    f"Ill-conditioned matrix (cond≈{cond:.1e}), "
                    "using pseudo-inverse for numerical stability",
                    stacklevel=2,
                )
                return np.asarray(linalg.pinv(A))
            return result
        except (linalg.LinAlgError, ValueError):
            warnings.warn(
                "Singular matrix encountered, using pseudo-inverse",
                stacklevel=2,
            )
            return np.asarray(linalg.pinv(A))

    def _check_convergence(self, beta_old: np.ndarray, beta_new: np.ndarray) -> bool:
        """Check convergence of iterative methods."""
        diff = np.max(np.abs(beta_new - beta_old))
        return bool(diff < self.tol)

    def _get_valid_mask(
        self,
        y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        min_instruments: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get mask of observations with sufficient valid data.

        Kept for backward compatibility.
        """
        y_valid = ~np.isnan(y).any(axis=1) if y.ndim > 1 else ~np.isnan(y)
        X_valid = ~np.isnan(X).any(axis=1)

        Z_notnan = ~np.isnan(Z)
        n_valid_instruments = Z_notnan.sum(axis=1)

        if min_instruments is None:
            k = X.shape[1] if X.ndim > 1 else 1
            min_instruments = k + 1

        Z_valid = n_valid_instruments >= min_instruments

        return np.asarray(y_valid & X_valid & Z_valid)
