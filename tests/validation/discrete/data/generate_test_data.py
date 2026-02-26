"""
Generate synthetic test datasets for validation against R.
"""

import numpy as np
import pandas as pd

np.random.seed(42)


def generate_binary_panel_data(n_entities=100, n_time=10):
    """Generate panel data for binary choice models."""
    # Create panel structure
    entity = np.repeat(np.arange(n_entities), n_time)
    time = np.tile(np.arange(n_time), n_entities)

    # Generate covariates
    x1 = np.random.randn(n_entities * n_time)
    x2 = np.random.randn(n_entities * n_time)

    # Generate entity-specific effects
    alpha_i = np.random.randn(n_entities)
    alpha = np.repeat(alpha_i, n_time)

    # Generate latent variable
    y_star = -1.5 + 0.8 * x1 + 0.5 * x2 + 0.3 * alpha + np.random.randn(n_entities * n_time)

    # Binary outcome
    y = (y_star > 0).astype(int)

    # Create DataFrame
    df = pd.DataFrame({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2})

    return df


def generate_count_panel_data(n_entities=100, n_time=10):
    """Generate panel data for count models."""
    entity = np.repeat(np.arange(n_entities), n_time)
    time = np.tile(np.arange(n_time), n_entities)

    # Generate covariates
    x1 = np.random.uniform(0, 2, n_entities * n_time)
    x2 = np.random.uniform(-1, 1, n_entities * n_time)

    # Entity effects
    alpha_i = np.random.gamma(2, 0.5, n_entities)
    alpha = np.repeat(alpha_i, n_time)

    # Lambda for Poisson
    lambda_it = np.exp(0.5 + 0.6 * x1 + 0.3 * x2 + 0.2 * np.log(alpha))

    # Generate count outcome
    y = np.random.poisson(lambda_it)

    df = pd.DataFrame({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2})

    return df


def generate_censored_panel_data(n_entities=100, n_time=10):
    """Generate panel data for censored models (Tobit)."""
    entity = np.repeat(np.arange(n_entities), n_time)
    time = np.tile(np.arange(n_time), n_entities)

    # Generate covariates
    x1 = np.random.randn(n_entities * n_time)
    x2 = np.random.randn(n_entities * n_time)

    # Entity effects
    alpha_i = np.random.randn(n_entities)
    alpha = np.repeat(alpha_i, n_time)

    # Latent variable
    y_star = 2 + 1.2 * x1 + 0.8 * x2 + 0.5 * alpha + np.random.randn(n_entities * n_time)

    # Censoring at 0
    y = np.maximum(0, y_star)

    df = pd.DataFrame(
        {
            "entity": entity,
            "time": time,
            "y": y,
            "x1": x1,
            "x2": x2,
            "censored": (y == 0).astype(int),
        }
    )

    return df


def generate_ordered_panel_data(n_entities=100, n_time=10, n_categories=4):
    """Generate panel data for ordered choice models."""
    entity = np.repeat(np.arange(n_entities), n_time)
    time = np.tile(np.arange(n_time), n_entities)

    # Generate covariates
    x1 = np.random.randn(n_entities * n_time)
    x2 = np.random.randn(n_entities * n_time)

    # Entity effects
    alpha_i = np.random.randn(n_entities) * 0.5
    alpha = np.repeat(alpha_i, n_time)

    # Latent variable
    y_star = 0.7 * x1 + 0.5 * x2 + alpha + np.random.randn(n_entities * n_time)

    # Create ordered categories with thresholds
    thresholds = np.array([-1.5, 0, 1.5])
    y = np.digitize(y_star, thresholds)

    df = pd.DataFrame({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2})

    return df


if __name__ == "__main__":
    # Generate all datasets
    binary_data = generate_binary_panel_data(200, 15)
    count_data = generate_count_panel_data(150, 12)
    censored_data = generate_censored_panel_data(100, 10)
    ordered_data = generate_ordered_panel_data(120, 10)

    # Save to CSV
    binary_data.to_csv("panel_binary.csv", index=False)
    count_data.to_csv("panel_count.csv", index=False)
    censored_data.to_csv("panel_censored.csv", index=False)
    ordered_data.to_csv("panel_ordered.csv", index=False)

    print("Test datasets generated successfully!")
    print(f"Binary data: {binary_data.shape}")
    print(f"Count data: {count_data.shape}")
    print(f"Censored data: {censored_data.shape}")
    print(f"Ordered data: {ordered_data.shape}")
