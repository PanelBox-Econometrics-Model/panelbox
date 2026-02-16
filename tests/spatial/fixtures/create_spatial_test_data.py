# tests/spatial/fixtures/create_spatial_test_data.py

from pathlib import Path

import numpy as np
import pandas as pd


def create_spatial_panel_data(
    N: int = 50, T: int = 10, rho: float = 0.4, lambda_: float = 0.3, seed: int = 42
) -> tuple:
    """
    Create synthetic spatial panel data with known parameters.

    Parameters
    ----------
    N : int
        Number of entities (cross-sectional units)
    T : int
        Number of time periods
    rho : float
        Spatial lag parameter
    lambda_ : float
        Spatial error parameter
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (data_df, W, true_params)
        - data_df: DataFrame with columns [entity, time, y, x1, x2, x3]
        - W: Spatial weight matrix (N x N)
        - true_params: Dictionary with true parameter values
    """
    np.random.seed(seed)

    # ========================================
    # 1. Create Spatial Weight Matrix
    # ========================================
    # Contiguity-like structure (circular lattice)
    W = np.zeros((N, N))

    for i in range(N):
        # Connect to nearest neighbors (circular)
        neighbors = [(i - 1) % N, (i + 1) % N]

        # Add second-order neighbors for richer structure
        if N > 10:
            neighbors.extend([(i - 2) % N, (i + 2) % N])

        for j in neighbors:
            W[i, j] = 1.0

    # Row normalize
    row_sums = W.sum(axis=1, keepdims=True)
    W = W / row_sums

    # ========================================
    # 2. True Parameters
    # ========================================
    beta_true = np.array([1.5, -0.8, 0.5])
    sigma_alpha = 1.0  # Random effect std
    sigma_eps = 0.5  # Idiosyncratic error std

    true_params = {
        "rho": rho,
        "lambda": lambda_,
        "beta": beta_true,
        "sigma_alpha": sigma_alpha,
        "sigma_eps": sigma_eps,
        "N": N,
        "T": T,
    }

    # ========================================
    # 3. Generate Random Effects (alpha_i)
    # ========================================
    alpha = np.random.normal(0, sigma_alpha, N)

    # ========================================
    # 4. Generate Data by Time Period
    # ========================================
    data = []

    for t in range(T):
        # Independent variables (time-varying)
        X = np.column_stack(
            [
                np.random.normal(5, 2, N),  # x1
                np.random.normal(3, 1, N),  # x2
                np.random.uniform(0, 10, N),  # x3
            ]
        )

        # Idiosyncratic error
        eps = np.random.normal(0, sigma_eps, N)

        # Generate y with spatial lag
        # Model: y = (I - rho*W)^{-1} * (X*beta + alpha + eps)
        I_N = np.eye(N)
        I_rhoW_inv = np.linalg.inv(I_N - rho * W)

        y = I_rhoW_inv @ (X @ beta_true + alpha + eps)

        # Store data
        for i in range(N):
            data.append(
                {"entity": i, "time": t, "y": y[i], "x1": X[i, 0], "x2": X[i, 1], "x3": X[i, 2]}
            )

    df = pd.DataFrame(data)

    return df, W, true_params


def save_test_data(output_dir: str = None):
    """
    Generate and save test data to CSV files.

    Parameters
    ----------
    output_dir : str
        Directory to save files. Defaults to current script directory.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    df, W, true_params = create_spatial_panel_data(N=50, T=10, rho=0.4, lambda_=0.3, seed=42)

    # Save data
    df.to_csv(output_dir / "spatial_test_data.csv", index=False)
    np.savetxt(output_dir / "spatial_weights.csv", W, delimiter=",")

    # Save true parameters
    import json

    with open(output_dir / "true_params.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        params_json = {
            "rho": true_params["rho"],
            "lambda": true_params["lambda"],
            "beta": true_params["beta"].tolist(),
            "sigma_alpha": true_params["sigma_alpha"],
            "sigma_eps": true_params["sigma_eps"],
            "N": true_params["N"],
            "T": true_params["T"],
        }
        json.dump(params_json, f, indent=2)

    print(f"Test data saved to {output_dir}")
    print(f"  - spatial_test_data.csv: {len(df)} rows")
    print(f"  - spatial_weights.csv: {W.shape} matrix")
    print(f"  - true_params.json")


if __name__ == "__main__":
    save_test_data()
