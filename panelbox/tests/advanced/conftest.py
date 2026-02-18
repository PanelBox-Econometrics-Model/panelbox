"""
Common fixtures for advanced methods tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def panel_data(seed):
    """Generate basic panel data."""
    n_entities = 50
    n_periods = 10
    n = n_entities * n_periods

    data = pd.DataFrame(
        {
            "entity": np.repeat(range(n_entities), n_periods),
            "time": np.tile(range(n_periods), n_entities),
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "z1": np.random.randn(n),
            "z2": np.random.randn(n),
        }
    )

    # Entity effects
    entity_effects = np.random.randn(n_entities)
    data["entity_effect"] = data["entity"].map(lambda x: entity_effects[x])

    # Generate y
    data["y"] = 1 + 2 * data["x1"] + 1.5 * data["x2"] + data["entity_effect"] + np.random.randn(n)

    return data


@pytest.fixture
def gmm_data(seed):
    """Generate data for GMM tests with instruments."""
    n = 500

    # Instruments
    z1 = np.random.randn(n)
    z2 = np.random.randn(n)
    z3 = np.random.randn(n)

    # Endogenous regressors
    x1 = 0.5 * z1 + 0.3 * z2 + np.random.randn(n) * 0.5
    x2 = 0.4 * z2 + 0.3 * z3 + np.random.randn(n) * 0.5

    # Outcome
    y = 1 + 2 * x1 + 1.5 * x2 + np.random.randn(n)

    data = pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "entity": np.repeat(range(50), 10),
            "time": np.tile(range(10), 50),
        }
    )

    return data


@pytest.fixture
def selection_data(seed):
    """Generate data for Heckman selection models."""
    n = 1000

    z1 = np.random.randn(n)
    z2 = np.random.randn(n)
    x1 = np.random.randn(n)

    # Selection
    s_star = 0.5 + 0.8 * z1 + 0.4 * z2 + np.random.randn(n)
    s = (s_star > 0).astype(int)

    # Outcome with correlated errors
    rho = 0.5
    u = rho * (s_star - (0.5 + 0.8 * z1 + 0.4 * z2)) + np.sqrt(1 - rho**2) * np.random.randn(n)
    y_star = 1 + 2 * x1 + u
    y = np.where(s == 1, y_star, np.nan)

    data = pd.DataFrame(
        {
            "s": s,
            "y": y,
            "z1": z1,
            "z2": z2,
            "x1": x1,
            "entity": np.repeat(range(100), 10),
            "time": np.tile(range(10), 100),
        }
    )

    return data


@pytest.fixture
def choice_data(seed):
    """Generate data for discrete choice models."""
    n_trips = 100
    modes = ["car", "bus", "train"]

    data = []
    for trip in range(n_trips):
        costs = np.array(
            [10 + np.random.randn() * 2, 5 + np.random.randn() * 1, 8 + np.random.randn() * 1.5]
        )
        times = np.array(
            [30 + np.random.randn() * 5, 45 + np.random.randn() * 10, 25 + np.random.randn() * 5]
        )

        # True utility
        beta_cost, beta_time = -0.3, -0.05
        utilities = beta_cost * costs + beta_time * times
        utilities += np.random.gumbel(size=3)
        chosen_idx = np.argmax(utilities)

        for j, mode in enumerate(modes):
            data.append(
                {
                    "trip_id": trip,
                    "mode": mode,
                    "chosen": 1 if j == chosen_idx else 0,
                    "cost": costs[j],
                    "time": times[j],
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def multinomial_data(seed):
    """Generate data for multinomial logit."""
    n = 300
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # Utilities for 3 alternatives
    u1 = np.random.gumbel(size=n)
    u2 = 0.5 + 1.0 * x1 + 0.3 * x2 + np.random.gumbel(size=n)
    u3 = 0.8 + 0.5 * x1 + 0.8 * x2 + np.random.gumbel(size=n)

    utilities = np.column_stack([u1, u2, u3])
    choice = np.argmax(utilities, axis=1)

    return pd.DataFrame(
        {
            "choice": choice,
            "x1": x1,
            "x2": x2,
            "entity": np.repeat(range(30), 10),
            "time": np.tile(range(10), 30),
        }
    )
