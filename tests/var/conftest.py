"""
Fixtures and helper functions for Panel VAR tests.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var import PanelVARData


@pytest.fixture
def simple_panel_data():
    """
    Create simple panel data for testing.

    Returns 5 entities × 20 periods × 2 variables.
    """
    np.random.seed(42)
    n_entities = 5
    n_periods = 20

    data = []
    for i in range(n_entities):
        for t in range(n_periods):
            row = {
                "entity": f"E{i}",
                "time": t,
                "y1": np.random.randn(),
                "y2": np.random.randn(),
            }
            data.append(row)

    return pd.DataFrame(data)


@pytest.fixture
def var_dgp_data():
    """
    Create data from a known VAR(2) DGP with stable coefficients.

    Returns
    -------
    df : pd.DataFrame
        Simulated panel data
    true_A : dict
        True coefficient matrices {'A1': np.ndarray, 'A2': np.ndarray}
    """
    np.random.seed(123)
    n_entities = 10
    n_periods = 50
    K = 2  # Two variables

    # True VAR(2) coefficients (stable)
    A1 = np.array([[0.5, 0.1], [0.2, 0.4]])
    A2 = np.array([[0.1, 0.05], [0.05, 0.1]])

    # Residual covariance
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    Sigma_chol = np.linalg.cholesky(Sigma)

    data = []

    for i in range(n_entities):
        # Initial values
        y_history = [np.random.randn(K) * 0.5 for _ in range(2)]

        # Generate time series
        for t in range(n_periods):
            # VAR(2) process
            eps = Sigma_chol @ np.random.randn(K)
            y_new = A1 @ y_history[-1] + A2 @ y_history[-2] + eps

            y_history.append(y_new)

            # Store in dataframe
            row = {"entity": i, "time": t, "y1": y_new[0], "y2": y_new[1]}
            data.append(row)

    df = pd.DataFrame(data)
    true_params = {"A1": A1, "A2": A2, "Sigma": Sigma}

    return df, true_params


def generate_stable_var_data(n_entities=10, n_periods=50, K=2, p=2, seed=None):
    """
    Generate data from a stable VAR(p) process.

    Parameters
    ----------
    n_entities : int
        Number of entities
    n_periods : int
        Number of time periods
    K : int
        Number of variables
    p : int
        Number of lags
    seed : int, optional
        Random seed

    Returns
    -------
    df : pd.DataFrame
        Generated panel data
    true_params : dict
        True coefficient matrices
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random stable coefficients
    A_matrices = []
    for lag in range(p):
        # Small random coefficients
        A_l = np.random.randn(K, K) * 0.2 / (lag + 1)
        A_matrices.append(A_l)

    # Residual covariance (identity for simplicity)
    Sigma = np.eye(K)
    Sigma_chol = np.linalg.cholesky(Sigma)

    data = []

    for entity in range(n_entities):
        # Initial values
        y_history = [np.random.randn(K) * 0.1 for _ in range(p)]

        # Generate time series
        for t in range(n_periods):
            # VAR(p) process
            y_new = np.zeros(K)
            for lag in range(p):
                y_new += A_matrices[lag] @ y_history[-(lag + 1)]

            # Add noise
            eps = Sigma_chol @ np.random.randn(K)
            y_new += eps

            y_history.append(y_new)

            # Store in dataframe
            row = {"entity": entity, "time": t}
            for k in range(K):
                row[f"y{k+1}"] = y_new[k]
            data.append(row)

    df = pd.DataFrame(data)

    true_params = {"A_matrices": A_matrices, "Sigma": Sigma}

    return df, true_params
