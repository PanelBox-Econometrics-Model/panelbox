"""
Create test datasets for quantile regression validation.

These datasets will be used by both Python (PanelBox) and R (quantreg/rqpd)
to ensure consistent comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def create_simple_panel(N: int = 100, T: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Create simple balanced panel with known DGP.

    Model:
        y_it = 1 + 2*x1_it - 1.5*x2_it + 0.5*x3_it + alpha_i + epsilon_it

    where:
        - alpha_i ~ N(0, 1) : entity fixed effects
        - epsilon_it ~ (1 + 0.5*x1_it) * N(0, 1) : heteroskedastic errors
    """
    np.random.seed(seed)

    # Entity fixed effects
    alpha = np.random.normal(0, 1, N)

    data = []
    for i in range(N):
        for t in range(T):
            # Covariates
            x1 = np.random.normal(5, 2)
            x2 = np.random.normal(3, 1)
            x3 = np.random.uniform(0, 10)

            # Heteroskedastic error
            sigma = 1 + 0.5 * x1
            epsilon = sigma * np.random.normal(0, 1)

            # Outcome
            y = 1 + 2 * x1 - 1.5 * x2 + 0.5 * x3 + alpha[i] + epsilon

            data.append({"entity": i, "time": t, "y": y, "x1": x1, "x2": x2, "x3": x3})

    return pd.DataFrame(data)


def create_location_scale_panel(N: int = 100, T: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Create panel following location-scale model for MSS (2019) validation.

    Model:
        y_it = X'α + σ(X) * ε_it

    where:
        - α = [1, 2, -1] : location parameters
        - log(σ) = X'γ with γ = [0.5, 0.3, -0.2] : scale parameters
        - ε_it ~ N(0, 1)
    """
    np.random.seed(seed)

    alpha = np.array([1, 2, -1])  # Location
    gamma = np.array([0.5, 0.3, -0.2])  # Log-scale

    data = []
    for i in range(N):
        for t in range(T):
            # Covariates (including constant)
            x = np.array([1, np.random.normal(5, 2), np.random.uniform(0, 10)])

            # Location and scale
            location = x @ alpha
            log_scale = x @ gamma
            scale = np.exp(log_scale)

            # Outcome
            epsilon = np.random.normal(0, 1)
            y = location + scale * epsilon

            data.append({"entity": i, "time": t, "y": y, "x1": x[1], "x2": x[2]})

    return pd.DataFrame(data)


def create_quantile_heterogeneity_panel(N: int = 100, T: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Create panel where true coefficients vary across quantiles.

    Model:
        Q_y(τ|X) = β₀(τ) + β₁(τ)*x1 + β₂(τ)*x2 + αᵢ

    where:
        - β₁(τ) = 2 + 0.5*Φ⁻¹(τ)  (increases with τ)
        - β₂(τ) = -1.5  (constant)
    """
    np.random.seed(seed)
    from scipy.stats import norm

    # Entity fixed effects
    alpha = np.random.normal(0, 1, N)

    data = []
    for i in range(N):
        for t in range(T):
            # Covariates
            x1 = np.random.normal(5, 2)
            x2 = np.random.normal(3, 1)

            # Draw a "quantile" u ~ Uniform(0,1)
            u = np.random.uniform(0, 1)

            # Quantile-varying coefficient
            beta1_tau = 2 + 0.5 * norm.ppf(u)
            beta2_tau = -1.5

            # Outcome
            y = 1 + beta1_tau * x1 + beta2_tau * x2 + alpha[i]

            data.append({"entity": i, "time": t, "y": y, "x1": x1, "x2": x2})

    return pd.DataFrame(data)


def save_datasets(output_dir: str = "tests/validation/quantile/fixtures"):
    """Save all test datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Simple panel
    df_simple = create_simple_panel()
    df_simple.to_csv(output_path / "test_data_simple.csv", index=False)
    print(f"Saved simple panel: {len(df_simple)} obs")

    # Location-scale panel
    df_ls = create_location_scale_panel()
    df_ls.to_csv(output_path / "test_data_location_scale.csv", index=False)
    print(f"Saved location-scale panel: {len(df_ls)} obs")

    # Quantile heterogeneity panel
    df_het = create_quantile_heterogeneity_panel()
    df_het.to_csv(output_path / "test_data_heterogeneity.csv", index=False)
    print(f"Saved heterogeneity panel: {len(df_het)} obs")

    # Save true parameters for validation
    true_params = {
        "simple": {"intercept": 1.0, "x1": 2.0, "x2": -1.5, "x3": 0.5, "sigma_alpha": 1.0},
        "location_scale": {"location": [1, 2, -1], "scale": [0.5, 0.3, -0.2]},
    }

    import json

    with open(output_path / "true_parameters.json", "w") as f:
        json.dump(true_params, f, indent=2)

    print(f"\nAll datasets saved to {output_path}")


if __name__ == "__main__":
    save_datasets()
