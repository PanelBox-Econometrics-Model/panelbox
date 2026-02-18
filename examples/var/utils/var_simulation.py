"""
VAR process simulation for pedagogical demonstrations.

Functions:
- simulate_var: Simulate a stationary VAR(p) process
- check_stability: Check VAR stability via companion matrix eigenvalues
- simulate_panel_var: Simulate panel VAR with entity fixed effects
- companion_matrix: Construct companion matrix from A matrices
- theoretical_irf: Compute theoretical (population) IRF for known DGP
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd


def companion_matrix(A_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Construct the companion matrix from a list of A matrices.

    For VAR(p) with K variables, the companion matrix is Kp x Kp:
    [[A_1, A_2, ..., A_p],
     [I_K,  0,  ...,  0 ],
     [ 0,  I_K, ...,  0 ],
     [ 0,   0,  ...,  0 ]]

    Parameters
    ----------
    A_matrices : list of np.ndarray
        List of K x K coefficient matrices [A_1, ..., A_p].

    Returns
    -------
    np.ndarray of shape (Kp, Kp)

    Raises
    ------
    ValueError
        If A_matrices is empty or matrices are not square / consistent sizes.
    """
    if not A_matrices:
        raise ValueError("A_matrices must contain at least one matrix.")

    K = A_matrices[0].shape[0]
    p = len(A_matrices)

    for i, A in enumerate(A_matrices):
        if A.shape != (K, K):
            raise ValueError(f"A_matrices[{i}] has shape {A.shape}, expected ({K}, {K}).")

    Kp = K * p

    # VAR(1): companion matrix is just A_1 itself
    if p == 1:
        return A_matrices[0].copy()

    # VAR(p > 1): build the full companion matrix
    C = np.zeros((Kp, Kp))

    # First block row: [A_1, A_2, ..., A_p]
    for l in range(p):
        C[0:K, l * K : (l + 1) * K] = A_matrices[l]

    # Remaining block rows: identity blocks on the sub-diagonal
    # Row block i (i = 1, ..., p-1) has I_K at column block (i-1)
    for i in range(1, p):
        C[i * K : (i + 1) * K, (i - 1) * K : i * K] = np.eye(K)

    return C


def check_stability(A_matrices: List[np.ndarray]) -> dict:
    """
    Check VAR stability by computing eigenvalues of companion matrix.

    A VAR(p) process is stable (stationary) if and only if all eigenvalues
    of the companion matrix lie strictly inside the unit circle.

    Parameters
    ----------
    A_matrices : list of np.ndarray
        List of K x K coefficient matrices.

    Returns
    -------
    dict
        Keys:
        - 'is_stable' (bool): True if max eigenvalue modulus < 1.
        - 'max_modulus' (float): Maximum modulus among eigenvalues.
        - 'eigenvalues' (np.ndarray): All eigenvalues of the companion matrix.
        - 'companion' (np.ndarray): The companion matrix itself.
    """
    C = companion_matrix(A_matrices)
    eigenvalues = np.linalg.eigvals(C)
    moduli = np.abs(eigenvalues)
    max_modulus = float(np.max(moduli))

    return {
        "is_stable": max_modulus < 1.0,
        "max_modulus": max_modulus,
        "eigenvalues": eigenvalues,
        "companion": C,
    }


def simulate_var(
    A_matrices: List[np.ndarray],
    Sigma: np.ndarray,
    n_obs: int = 200,
    burn_in: int = 50,
    intercept: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate a stationary VAR(p) process.

    Parameters
    ----------
    A_matrices : list of np.ndarray
        List of K x K coefficient matrices [A_1, ..., A_p].
    Sigma : np.ndarray
        K x K error covariance matrix.
    n_obs : int, default 200
        Number of observations to generate (after burn-in).
    burn_in : int, default 50
        Number of initial observations to discard.
    intercept : np.ndarray, optional
        K-vector of intercepts (default: zeros).
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray of shape (n_obs, K)

    Raises
    ------
    ValueError
        If the process is not stationary (max eigenvalue modulus >= 1).
    ValueError
        If Sigma is not a valid K x K covariance matrix.
    """
    # Validate inputs
    if not A_matrices:
        raise ValueError("A_matrices must contain at least one matrix.")

    K = A_matrices[0].shape[0]
    p = len(A_matrices)

    if Sigma.shape != (K, K):
        raise ValueError(f"Sigma has shape {Sigma.shape}, expected ({K}, {K}).")

    # Check stationarity
    stability = check_stability(A_matrices)
    if not stability["is_stable"]:
        raise ValueError(
            f"VAR process is not stationary. Maximum eigenvalue modulus = "
            f"{stability['max_modulus']:.6f} (must be < 1)."
        )

    # Default intercept: zero vector
    if intercept is None:
        intercept = np.zeros(K)
    else:
        intercept = np.asarray(intercept, dtype=float).ravel()
        if intercept.shape[0] != K:
            raise ValueError(f"intercept has length {intercept.shape[0]}, expected {K}.")

    # Set random seed
    rng = np.random.default_rng(seed)

    # Total length of the simulation (including initial conditions and burn-in)
    T_total = n_obs + burn_in + p

    # Generate innovations: shape (T_total, K)
    eps = rng.multivariate_normal(mean=np.zeros(K), cov=Sigma, size=T_total)

    # Initialize y with zeros
    y = np.zeros((T_total, K))

    # Simulate the VAR(p) recursion
    for t in range(p, T_total):
        y[t] = intercept.copy()
        for l in range(1, p + 1):
            y[t] += A_matrices[l - 1] @ y[t - l]
        y[t] += eps[t]

    # Discard initial conditions (p values) and burn-in
    return y[burn_in + p :]


def simulate_panel_var(
    A_matrices: List[np.ndarray],
    Sigma: np.ndarray,
    n_entities: int,
    n_periods: int,
    fixed_effects_std: float = 1.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate panel VAR with entity fixed effects.

    Each entity i has a fixed effect mu_i drawn from N(0, fixed_effects_std * I_K)
    that acts as the intercept of the VAR process for that entity.

    Parameters
    ----------
    A_matrices : list of np.ndarray
        List of K x K coefficient matrices.
    Sigma : np.ndarray
        K x K error covariance matrix.
    n_entities : int
        Number of panel entities.
    n_periods : int
        Number of time periods per entity.
    fixed_effects_std : float, default 1.0
        Standard deviation of entity fixed effects.
    seed : int, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        Long-format with columns: entity, time, y_0, y_1, ..., y_{K-1}.

    Raises
    ------
    ValueError
        If the process is not stationary or inputs are invalid.
    """
    if n_entities < 1:
        raise ValueError("n_entities must be >= 1.")
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1.")
    if fixed_effects_std < 0:
        raise ValueError("fixed_effects_std must be >= 0.")

    K = A_matrices[0].shape[0]
    rng = np.random.default_rng(seed)

    frames = []
    for i in range(n_entities):
        # Draw entity-specific fixed effect
        mu_i = rng.normal(loc=0.0, scale=fixed_effects_std, size=K)

        # Generate a per-entity seed from the parent rng for reproducibility
        entity_seed = int(rng.integers(0, 2**31))

        # Simulate the VAR for this entity with mu_i as the intercept
        y_i = simulate_var(
            A_matrices=A_matrices,
            Sigma=Sigma,
            n_obs=n_periods,
            burn_in=50,
            intercept=mu_i,
            seed=entity_seed,
        )

        # Build a DataFrame for this entity
        df_i = pd.DataFrame(y_i, columns=[f"y_{k}" for k in range(K)])
        df_i.insert(0, "entity", i)
        df_i.insert(1, "time", np.arange(n_periods))

        frames.append(df_i)

    panel = pd.concat(frames, ignore_index=True)
    return panel


def theoretical_irf(
    A_matrices: List[np.ndarray], Sigma: np.ndarray, periods: int = 20, method: str = "cholesky"
) -> np.ndarray:
    """
    Compute theoretical (population) IRF for known DGP.

    Uses the moving-average (MA) representation of the VAR process.
    The MA coefficients are computed recursively:
        Phi_0 = I_K
        Phi_h = sum_{l=1}^{min(h,p)} A_l @ Phi_{h-l}    for h >= 1

    Then the orthogonalized IRF is obtained by multiplying Phi_h by
    an identification matrix derived from Sigma.

    Parameters
    ----------
    A_matrices : list of np.ndarray
        List of K x K coefficient matrices.
    Sigma : np.ndarray
        K x K error covariance matrix.
    periods : int, default 20
        Number of IRF periods.
    method : str, default 'cholesky'
        Identification method:
        - 'cholesky': Uses the lower Cholesky factor P of Sigma,
          so IRF[h] = Phi_h @ P. This corresponds to a recursive
          ordering identification.
        - 'generalized': Uses the Pesaran-Shin generalized IRF,
          where IRF[h, i, j] = (Phi_h @ Sigma @ e_j) / sqrt(Sigma[j,j]).
          This does not depend on variable ordering.

    Returns
    -------
    np.ndarray of shape (periods+1, K, K)
        IRF[h, i, j] = response of variable i at horizon h
        to a one-standard-deviation shock in variable j.

    Raises
    ------
    ValueError
        If method is not 'cholesky' or 'generalized'.
    ValueError
        If Sigma is not positive definite (for Cholesky method).
    """
    if method not in ("cholesky", "generalized"):
        raise ValueError(f"method must be 'cholesky' or 'generalized', got '{method}'.")

    if not A_matrices:
        raise ValueError("A_matrices must contain at least one matrix.")

    K = A_matrices[0].shape[0]
    p = len(A_matrices)

    if Sigma.shape != (K, K):
        raise ValueError(f"Sigma has shape {Sigma.shape}, expected ({K}, {K}).")

    # Step 1: Compute MA coefficients Phi_0, Phi_1, ..., Phi_{periods}
    Phi = np.zeros((periods + 1, K, K))
    Phi[0] = np.eye(K)

    for h in range(1, periods + 1):
        for l in range(1, min(h, p) + 1):
            Phi[h] += A_matrices[l - 1] @ Phi[h - l]

    # Step 2: Apply identification scheme
    irf = np.zeros((periods + 1, K, K))

    if method == "cholesky":
        # Lower Cholesky factor of Sigma
        try:
            P = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            raise ValueError("Sigma is not positive definite; Cholesky decomposition failed.")
        for h in range(periods + 1):
            irf[h] = Phi[h] @ P

    elif method == "generalized":
        # Generalized IRF (Pesaran & Shin, 1998)
        # IRF[h, :, j] = Phi_h @ Sigma @ e_j / sqrt(Sigma[j,j])
        for h in range(periods + 1):
            for j in range(K):
                e_j = np.zeros(K)
                e_j[j] = 1.0
                sigma_jj = Sigma[j, j]
                if sigma_jj <= 0:
                    raise ValueError(
                        f"Sigma[{j},{j}] = {sigma_jj} is not positive; "
                        f"cannot compute generalized IRF."
                    )
                irf[h, :, j] = (Phi[h] @ Sigma @ e_j) / np.sqrt(sigma_jj)

    return irf
