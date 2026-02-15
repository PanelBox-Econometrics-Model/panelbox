#!/usr/bin/env python3
"""
Generate validation data for spatial panel models.

This script generates synthetic data with known parameters for testing
spatial panel model implementations against R splm.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def generate_spatial_weights(n, w_type="rook", density=0.3, seed=None):
    """Generate spatial weight matrix."""
    if seed is not None:
        np.random.seed(seed)

    if w_type == "rook":
        # Simple chain structure (1D rook contiguity)
        W = np.zeros((n, n))
        for i in range(n - 1):
            W[i, i + 1] = 1
            W[i + 1, i] = 1

    elif w_type == "queen":
        # Grid-based queen contiguity
        grid_size = int(np.sqrt(n))
        if grid_size**2 != n:
            raise ValueError(f"n must be perfect square for queen, got {n}")

        W = np.zeros((n, n))
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j

                # Add all 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue

                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            nidx = ni * grid_size + nj
                            W[idx, nidx] = 1

    elif w_type == "random":
        # Random sparse matrix
        W = np.random.binomial(1, density, size=(n, n))
        W = (W + W.T) / 2  # Make symmetric
        np.fill_diagonal(W, 0)

    else:
        raise ValueError(f"Unknown type: {w_type}")

    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    return W


def generate_sar_data(n, t, rho, beta, W, sigma2=1.0, alpha_sd=1.0, seed=None):
    """Generate SAR panel data: y = ρWy + Xβ + α + ε."""
    if seed is not None:
        np.random.seed(seed)

    k = len(beta)

    # Fixed effects
    alpha = np.random.normal(0, alpha_sd, n)

    # Exogenous variables
    X = np.random.normal(0, 1, (n * t, k))

    # Errors
    epsilon = np.random.normal(0, np.sqrt(sigma2), n * t)

    # Panel structure
    entity = np.repeat(np.arange(n), t)
    time = np.tile(np.arange(t), n)

    # Generate y solving (I - ρW)y = Xβ + α + ε
    y = np.zeros(n * t)
    for period in range(t):
        idx_t = time == period
        X_t = X[idx_t]
        alpha_t = alpha
        epsilon_t = epsilon[idx_t]

        # Reduced form
        I_rhoW = np.eye(n) - rho * W
        I_rhoW_inv = np.linalg.inv(I_rhoW)

        y_t = I_rhoW_inv @ (X_t @ beta + alpha_t + epsilon_t)
        y[idx_t] = y_t

    # Create DataFrame
    data = pd.DataFrame({"entity": entity, "time": time, "y": y})

    for j in range(k):
        data[f"x{j+1}"] = X[:, j]

    return data


def generate_sem_data(n, t, lambda_, beta, W, sigma2=1.0, alpha_sd=1.0, seed=None):
    """Generate SEM panel data: y = Xβ + α + u, u = λWu + ε."""
    if seed is not None:
        np.random.seed(seed)

    k = len(beta)

    # Fixed effects
    alpha = np.random.normal(0, alpha_sd, n)

    # Exogenous variables
    X = np.random.normal(0, 1, (n * t, k))

    # Innovations
    epsilon = np.random.normal(0, np.sqrt(sigma2), n * t)

    # Panel structure
    entity = np.repeat(np.arange(n), t)
    time = np.tile(np.arange(t), n)

    # Generate spatially correlated errors
    y = np.zeros(n * t)
    for period in range(t):
        idx_t = time == period
        X_t = X[idx_t]
        alpha_t = alpha
        epsilon_t = epsilon[idx_t]

        # Spatial error: u = (I - λW)^{-1}ε
        I_lambdaW = np.eye(n) - lambda_ * W
        I_lambdaW_inv = np.linalg.inv(I_lambdaW)
        u_t = I_lambdaW_inv @ epsilon_t

        # Generate y
        y_t = X_t @ beta + alpha_t + u_t
        y[idx_t] = y_t

    # Create DataFrame
    data = pd.DataFrame({"entity": entity, "time": time, "y": y})

    for j in range(k):
        data[f"x{j+1}"] = X[:, j]

    return data


def generate_validation_datasets():
    """Generate all validation datasets."""

    # Create output directory
    data_dir = Path("tests/validation_spatial/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Generating validation datasets...")
    print("=" * 50)

    # Test scenarios
    scenarios = [
        # Small datasets
        {
            "name": "synthetic_n25_t10",
            "n": 25,
            "t": 10,
            "w_type": "rook",
            "rho": 0.4,
            "lambda": 0.4,
            "beta": [1.0, -0.5],
            "seed": 42,
        },
        # Medium datasets with grid structure
        {
            "name": "synthetic_n49_t15",
            "n": 49,
            "t": 15,
            "w_type": "queen",
            "rho": 0.3,
            "lambda": 0.3,
            "beta": [1.5, -0.8],
            "seed": 456,
        },
        # Larger datasets
        {
            "name": "synthetic_n100_t20",
            "n": 100,
            "t": 20,
            "w_type": "queen",
            "rho": 0.5,
            "lambda": 0.5,
            "beta": [2.0, -1.0, 0.5],
            "seed": 123,
        },
    ]

    for scenario in scenarios:
        print(f"\nGenerating: {scenario['name']}")
        print(f"  Dimensions: N={scenario['n']}, T={scenario['t']}")
        print(f"  Weight type: {scenario['w_type']}")

        # Generate W matrix
        W = generate_spatial_weights(
            scenario["n"], w_type=scenario["w_type"], seed=scenario["seed"]
        )

        # Generate SAR data
        sar_data = generate_sar_data(
            n=scenario["n"],
            t=scenario["t"],
            rho=scenario["rho"],
            beta=np.array(scenario["beta"]),
            W=W,
            seed=scenario["seed"],
        )

        # Generate SEM data
        sem_data = generate_sem_data(
            n=scenario["n"],
            t=scenario["t"],
            lambda_=scenario["lambda"],
            beta=np.array(scenario["beta"]),
            W=W,
            seed=scenario["seed"] + 1000,
        )

        # Save datasets
        sar_file = data_dir / f"sar_{scenario['name']}.csv"
        sem_file = data_dir / f"sem_{scenario['name']}.csv"
        w_file = data_dir / f"W_{scenario['name']}.npz"

        sar_data.to_csv(sar_file, index=False)
        sem_data.to_csv(sem_file, index=False)
        np.savez_compressed(w_file, W=W)

        # Save metadata
        metadata = {
            "name": scenario["name"],
            "n": scenario["n"],
            "t": scenario["t"],
            "w_type": scenario["w_type"],
            "true_params": {
                "rho": scenario["rho"],
                "lambda": scenario["lambda"],
                "beta": scenario["beta"],
            },
            "files": {
                "sar_data": str(sar_file.name),
                "sem_data": str(sem_file.name),
                "W_matrix": str(w_file.name),
            },
        }

        metadata_file = data_dir / f"metadata_{scenario['name']}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  SAR data: {sar_file.name}")
        print(f"  SEM data: {sem_file.name}")
        print(f"  W matrix: {w_file.name}")
        print(f"  Metadata: {metadata_file.name}")

    print("\n" + "=" * 50)
    print("All validation datasets generated successfully!")

    # Generate summary file
    summary = {
        "description": "Validation datasets for spatial panel models",
        "scenarios": scenarios,
        "notes": [
            "All datasets include fixed effects",
            "W matrices are row-normalized",
            "Data generated with known true parameters for validation",
        ],
    }

    summary_file = data_dir / "DATASET_SUMMARY.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    generate_validation_datasets()
