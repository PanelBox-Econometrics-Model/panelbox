"""
Generate synthetic panel data for validation against R implementations.
"""

import numpy as np
import pandas as pd


def generate_panel_binary_data(n_entities=500, n_time=10, seed=42):
    """
    Generate synthetic panel data for binary choice models.

    Parameters
    ----------
    n_entities : int
        Number of cross-sectional units
    n_time : int
        Number of time periods
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Panel data with binary outcome
    """
    np.random.seed(seed)

    # Entity and time indices
    entity = np.repeat(np.arange(1, n_entities + 1), n_time)
    time = np.tile(np.arange(1, n_time + 1), n_entities)

    # Generate entity-specific effects
    alpha_i = np.repeat(np.random.normal(0, 0.5, n_entities), n_time)

    # Generate covariates
    x1 = np.random.normal(0, 1, n_entities * n_time)
    x2 = np.random.normal(0, 1, n_entities * n_time)
    x3 = np.random.uniform(0, 1, n_entities * n_time)

    # Time-varying covariate with entity-specific trend
    entity_trend = np.repeat(np.random.normal(0, 0.2, n_entities), n_time)
    x4 = (
        0.1 * np.tile(np.arange(1, n_time + 1), n_entities)
        + entity_trend
        + np.random.normal(0, 0.5, n_entities * n_time)
    )

    # True parameters
    beta_1 = 0.5
    beta_2 = -0.3
    beta_3 = 0.8
    beta_4 = 0.2

    # Generate latent variable (Probit model)
    latent = (
        alpha_i
        + beta_1 * x1
        + beta_2 * x2
        + beta_3 * x3
        + beta_4 * x4
        + np.random.normal(0, 1, n_entities * n_time)
    )

    # Binary outcome
    y = (latent > 0).astype(int)

    # Create DataFrame
    data = pd.DataFrame(
        {"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4}
    )

    return data


def generate_panel_count_data(n_entities=500, n_time=10, seed=42):
    """
    Generate synthetic panel data for count models.

    Parameters
    ----------
    n_entities : int
        Number of cross-sectional units
    n_time : int
        Number of time periods
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with count outcome
    """
    np.random.seed(seed)

    # Entity and time indices
    entity = np.repeat(np.arange(1, n_entities + 1), n_time)
    time = np.tile(np.arange(1, n_time + 1), n_entities)

    # Generate entity-specific effects
    alpha_i = np.repeat(np.random.gamma(2, 0.5, n_entities), n_time)

    # Generate covariates
    x1 = np.random.normal(0, 1, n_entities * n_time)
    x2 = np.random.normal(0, 1, n_entities * n_time)
    x3 = np.random.uniform(0, 1, n_entities * n_time)

    # True parameters
    beta_1 = 0.3
    beta_2 = -0.2
    beta_3 = 0.5

    # Generate lambda for Poisson
    log_lambda = alpha_i + beta_1 * x1 + beta_2 * x2 + beta_3 * x3
    lambda_i = np.exp(log_lambda)

    # Generate count outcome (Poisson)
    y = np.random.poisson(lambda_i)

    # Create DataFrame
    data = pd.DataFrame({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2, "x3": x3})

    return data


def generate_panel_censored_data(n_entities=500, n_time=10, seed=42):
    """
    Generate synthetic panel data for censored models (Tobit).

    Parameters
    ----------
    n_entities : int
        Number of cross-sectional units
    n_time : int
        Number of time periods
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with censored outcome
    """
    np.random.seed(seed)

    # Entity and time indices
    entity = np.repeat(np.arange(1, n_entities + 1), n_time)
    time = np.tile(np.arange(1, n_time + 1), n_entities)

    # Generate entity-specific effects
    alpha_i = np.repeat(np.random.normal(0, 0.5, n_entities), n_time)

    # Generate covariates
    x1 = np.random.normal(0, 1, n_entities * n_time)
    x2 = np.random.normal(0, 1, n_entities * n_time)
    x3 = np.random.uniform(0, 1, n_entities * n_time)

    # True parameters
    beta_1 = 0.5
    beta_2 = -0.3
    beta_3 = 0.8

    # Generate latent variable
    y_star = (
        alpha_i
        + beta_1 * x1
        + beta_2 * x2
        + beta_3 * x3
        + np.random.normal(0, 1, n_entities * n_time)
    )

    # Censoring at 0 (left-censored Tobit)
    y = np.maximum(0, y_star)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "entity": entity,
            "time": time,
            "y": y,
            "y_censored": (y_star <= 0).astype(int),  # Censoring indicator
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )

    return data


def generate_panel_ordered_data(n_entities=500, n_time=10, n_categories=4, seed=42):
    """
    Generate synthetic panel data for ordered choice models.

    Parameters
    ----------
    n_entities : int
        Number of cross-sectional units
    n_time : int
        Number of time periods
    n_categories : int
        Number of ordered categories
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with ordered outcome
    """
    np.random.seed(seed)

    # Entity and time indices
    entity = np.repeat(np.arange(1, n_entities + 1), n_time)
    time = np.tile(np.arange(1, n_time + 1), n_entities)

    # Generate entity-specific effects
    alpha_i = np.repeat(np.random.normal(0, 0.3, n_entities), n_time)

    # Generate covariates
    x1 = np.random.normal(0, 1, n_entities * n_time)
    x2 = np.random.normal(0, 1, n_entities * n_time)
    x3 = np.random.uniform(0, 1, n_entities * n_time)

    # True parameters
    beta_1 = 0.4
    beta_2 = -0.3
    beta_3 = 0.6

    # Generate latent variable
    y_star = (
        alpha_i
        + beta_1 * x1
        + beta_2 * x2
        + beta_3 * x3
        + np.random.normal(0, 1, n_entities * n_time)
    )

    # Generate thresholds
    thresholds = np.linspace(-1, 1, n_categories - 1)

    # Map to ordered categories
    y = np.zeros(len(y_star), dtype=int)
    for i, threshold in enumerate(thresholds):
        y[y_star > threshold] = i + 1

    # Create DataFrame
    data = pd.DataFrame({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2, "x3": x3})

    return data


if __name__ == "__main__":
    # Generate all datasets
    print("Generating synthetic panel datasets for validation...")

    # Binary choice data
    binary_data = generate_panel_binary_data(n_entities=500, n_time=10)
    binary_data.to_csv("tests/validation/discrete/data/panel_binary.csv", index=False)
    print(
        f"Binary data: {binary_data.shape[0]} observations, {binary_data['y'].mean():.3f} positive rate"
    )

    # Count data
    count_data = generate_panel_count_data(n_entities=500, n_time=10)
    count_data.to_csv("tests/validation/count/data/panel_count.csv", index=False)
    print(
        f"Count data: {count_data.shape[0]} observations, mean={count_data['y'].mean():.2f}, var={count_data['y'].var():.2f}"
    )

    # Censored data
    censored_data = generate_panel_censored_data(n_entities=500, n_time=10)
    censored_data.to_csv("tests/validation/censored/data/panel_censored.csv", index=False)
    print(
        f"Censored data: {censored_data.shape[0]} observations, {censored_data['y_censored'].mean():.3f} censored rate"
    )

    # Ordered data
    ordered_data = generate_panel_ordered_data(n_entities=500, n_time=10, n_categories=4)
    ordered_data.to_csv("tests/validation/discrete/data/panel_ordered.csv", index=False)
    print(
        f"Ordered data: {ordered_data.shape[0]} observations, {ordered_data['y'].nunique()} categories"
    )

    print("\nAll datasets generated successfully!")
