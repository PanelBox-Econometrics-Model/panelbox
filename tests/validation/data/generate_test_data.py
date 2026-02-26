"""
Generate synthetic Panel VAR datasets for validation testing.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)


def generate_simple_pvar_data(N=50, T=20, K=3, p=2):
    """
    Generate simple Panel VAR(p) data with known DGP.

    DGP:
    y_{it} = A_1 * y_{i,t-1} + A_2 * y_{i,t-2} + e_{it}

    where:
    - A_1, A_2 are K x K coefficient matrices
    - e_{it} ~ N(0, Σ)
    """
    # Coefficient matrices (stable VAR)
    A1 = np.array([[0.5, 0.1, 0.0], [0.1, 0.4, 0.2], [0.0, 0.1, 0.3]])

    A2 = np.array([[0.2, 0.0, 0.1], [0.0, 0.2, 0.0], [0.1, 0.0, 0.2]])

    # Covariance matrix
    Sigma = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]])

    # Generate data
    data_list = []

    for i in range(N):
        # Initialize with zeros
        y = np.zeros((T + p, K))

        # Generate initial values
        y[:p] = np.random.multivariate_normal(np.zeros(K), Sigma, p)

        # Generate time series
        for t in range(p, T + p):
            y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + np.random.multivariate_normal(np.zeros(K), Sigma)

        # Remove burn-in period
        y = y[p:]

        # Create dataframe
        for t in range(T):
            data_list.append(
                {"entity": i + 1, "time": t + 1, "y1": y[t, 0], "y2": y[t, 1], "y3": y[t, 2]}
            )

    df = pd.DataFrame(data_list)
    return df, {"A1": A1, "A2": A2, "Sigma": Sigma}


def generate_love_zicchino_style_data(N=140, T=10):
    """
    Generate data similar to Love & Zicchino (2006) dataset.

    Variables: sales, inv, ar, debt
    """
    # Coefficient matrices (based on typical corporate finance relationships)
    A1 = np.array(
        [
            [0.6, 0.2, 0.1, -0.1],  # sales
            [0.3, 0.5, 0.0, 0.0],  # inventory
            [0.2, 0.1, 0.6, 0.0],  # accounts receivable
            [0.1, 0.0, 0.1, 0.7],  # debt
        ]
    )

    A2 = np.array(
        [[0.2, 0.0, 0.0, 0.0], [0.1, 0.2, 0.0, 0.0], [0.0, 0.1, 0.2, 0.0], [0.0, 0.0, 0.0, 0.2]]
    )

    # Covariance matrix
    Sigma = np.array(
        [[1.0, 0.4, 0.3, 0.2], [0.4, 1.0, 0.2, 0.1], [0.3, 0.2, 1.0, 0.1], [0.2, 0.1, 0.1, 1.0]]
    )

    K = 4
    p = 2

    data_list = []

    for i in range(N):
        # Initialize
        y = np.zeros((T + p, K))
        y[:p] = np.random.multivariate_normal(np.zeros(K), Sigma, p)

        # Generate
        for t in range(p, T + p):
            y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + np.random.multivariate_normal(np.zeros(K), Sigma)

        y = y[p:]

        # Create dataframe (with firm_id and year)
        for t in range(T):
            data_list.append(
                {
                    "firm_id": i + 1,
                    "year": 2000 + t,
                    "sales": y[t, 0],
                    "inv": y[t, 1],
                    "ar": y[t, 2],
                    "debt": y[t, 3],
                }
            )

    df = pd.DataFrame(data_list)
    return df, {"A1": A1, "A2": A2, "Sigma": Sigma}


def generate_unbalanced_panel(N=100, T_min=10, T_max=25):
    """
    Generate unbalanced panel dataset.
    """
    A1 = np.array([[0.5, 0.2], [0.1, 0.4]])

    A2 = np.array([[0.1, 0.0], [0.0, 0.1]])

    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

    K = 2
    p = 2

    data_list = []

    for i in range(N):
        # Random T for this entity
        T_i = np.random.randint(T_min, T_max + 1)

        y = np.zeros((T_i + p, K))
        y[:p] = np.random.multivariate_normal(np.zeros(K), Sigma, p)

        for t in range(p, T_i + p):
            y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + np.random.multivariate_normal(np.zeros(K), Sigma)

        y = y[p:]

        for t in range(T_i):
            data_list.append({"entity": i + 1, "time": t + 1, "y1": y[t, 0], "y2": y[t, 1]})

    df = pd.DataFrame(data_list)
    return df, {"A1": A1, "A2": A2, "Sigma": Sigma}


if __name__ == "__main__":
    # Create output directory
    output_dir = Path(__file__).parent

    # Generate and save simple dataset
    print("Generating simple Panel VAR dataset...")
    df_simple, params_simple = generate_simple_pvar_data()
    df_simple.to_csv(output_dir / "simple_pvar.csv", index=False)
    print("  Saved: simple_pvar.csv (N=50, T=20, K=3)")

    # Generate and save Love & Zicchino style dataset
    print("Generating Love & Zicchino style dataset...")
    df_lz, params_lz = generate_love_zicchino_style_data()
    df_lz.to_csv(output_dir / "love_zicchino_synthetic.csv", index=False)
    print("  Saved: love_zicchino_synthetic.csv (N=140, T=10, K=4)")

    # Generate and save unbalanced panel
    print("Generating unbalanced panel dataset...")
    df_unbal, params_unbal = generate_unbalanced_panel()
    df_unbal.to_csv(output_dir / "unbalanced_panel.csv", index=False)
    print("  Saved: unbalanced_panel.csv (N=100, T=min=10,max=25, K=2)")

    # Save true parameters
    import json

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    params_all = {
        "simple_pvar": {k: convert_to_serializable(v) for k, v in params_simple.items()},
        "love_zicchino_synthetic": {k: convert_to_serializable(v) for k, v in params_lz.items()},
        "unbalanced_panel": {k: convert_to_serializable(v) for k, v in params_unbal.items()},
    }

    with open(output_dir / "true_parameters.json", "w") as f:
        json.dump(params_all, f, indent=2)

    print("\n✓ All datasets generated successfully!")
    print(f"  Output directory: {output_dir}")
