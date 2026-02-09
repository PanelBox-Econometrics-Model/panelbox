"""
Pytest configuration and fixtures for panelbox tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def balanced_panel_data():
    """Create a balanced panel dataset for testing."""
    np.random.seed(42)

    n_entities = 10
    n_periods = 5
    n_obs = n_entities * n_periods

    data = pd.DataFrame(
        {
            "entity": np.repeat(range(1, n_entities + 1), n_periods),
            "time": np.tile(range(2020, 2020 + n_periods), n_entities),
            "y": np.random.randn(n_obs) * 10 + 100,
            "x1": np.random.randn(n_obs) * 5 + 50,
            "x2": np.random.randn(n_obs) * 3 + 30,
        }
    )

    return data


@pytest.fixture
def unbalanced_panel_data():
    """Create an unbalanced panel dataset for testing."""
    np.random.seed(42)

    # Entity 1: 5 periods
    # Entity 2: 4 periods
    # Entity 3: 3 periods
    data = pd.DataFrame(
        {
            "entity": [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            "time": [2020, 2021, 2022, 2023, 2024, 2020, 2021, 2022, 2023, 2020, 2021, 2022],
            "y": np.random.randn(12) * 10 + 100,
            "x1": np.random.randn(12) * 5 + 50,
            "x2": np.random.randn(12) * 3 + 30,
        }
    )

    return data


@pytest.fixture
def grunfeld_data():
    """
    Create a sample Grunfeld investment data for testing.

    This is a simplified version of the classic Grunfeld dataset
    used in many panel data examples.
    """
    np.random.seed(42)

    firms = ["GM", "CH", "GE", "WE", "US"]
    years = range(1935, 1955)
    n_obs = len(firms) * len(years)

    data = pd.DataFrame(
        {
            "firm": np.repeat(firms, len(years)),
            "year": np.tile(years, len(firms)),
            "invest": np.random.uniform(0, 500, n_obs),
            "value": np.random.uniform(500, 5000, n_obs),
            "capital": np.random.uniform(100, 3000, n_obs),
        }
    )

    return data


# ============================================================================
# Validation Test Fixtures
# ============================================================================


@pytest.fixture
def panel_with_ar1():
    """
    Create panel data with AR(1) serial correlation.

    Residuals follow: e_it = rho * e_i,t-1 + u_it
    with rho = 0.5
    """
    np.random.seed(123)

    n_entities = 50
    n_periods = 10
    rho = 0.5  # AR(1) coefficient

    # True parameters
    beta_0 = 10.0
    beta_1 = 2.0
    beta_2 = -1.5

    entities = np.repeat(range(1, n_entities + 1), n_periods)
    times = np.tile(range(1, n_periods + 1), n_entities)

    # Regressors
    x1 = np.random.randn(n_entities * n_periods) * 5 + 50
    x2 = np.random.randn(n_entities * n_periods) * 3 + 30

    # Entity fixed effects
    entity_effects = np.repeat(np.random.randn(n_entities) * 10, n_periods)

    # Generate AR(1) errors
    errors = np.zeros(n_entities * n_periods)
    innovations = np.random.randn(n_entities * n_periods) * 5

    for i in range(n_entities):
        start_idx = i * n_periods
        end_idx = (i + 1) * n_periods

        for t in range(n_periods):
            idx = start_idx + t
            if t == 0:
                errors[idx] = innovations[idx]
            else:
                errors[idx] = rho * errors[idx - 1] + innovations[idx]

    # Generate y with AR(1) errors
    y = beta_0 + beta_1 * x1 + beta_2 * x2 + entity_effects + errors

    data = pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    )

    return data


@pytest.fixture
def panel_with_heteroskedasticity():
    """
    Create panel data with groupwise heteroskedasticity.

    Error variance differs across entities.
    """
    np.random.seed(124)

    n_entities = 50
    n_periods = 10

    # True parameters
    beta_0 = 10.0
    beta_1 = 2.0
    beta_2 = -1.5

    entities = np.repeat(range(1, n_entities + 1), n_periods)
    times = np.tile(range(1, n_periods + 1), n_entities)

    # Regressors
    x1 = np.random.randn(n_entities * n_periods) * 5 + 50
    x2 = np.random.randn(n_entities * n_periods) * 3 + 30

    # Entity fixed effects
    entity_effects = np.repeat(np.random.randn(n_entities) * 10, n_periods)

    # Heteroskedastic errors (variance differs by entity)
    errors = np.zeros(n_entities * n_periods)
    for i in range(n_entities):
        # Each entity has different variance
        sigma_i = 1.0 + 0.5 * i  # Variance increases with entity index
        start_idx = i * n_periods
        end_idx = (i + 1) * n_periods
        errors[start_idx:end_idx] = np.random.randn(n_periods) * sigma_i

    # Generate y
    y = beta_0 + beta_1 * x1 + beta_2 * x2 + entity_effects + errors

    data = pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    )

    return data


@pytest.fixture
def panel_with_cross_sectional_dependence():
    """
    Create panel data with cross-sectional dependence.

    Errors are correlated across entities.
    """
    np.random.seed(125)

    n_entities = 30
    n_periods = 20

    # True parameters
    beta_0 = 10.0
    beta_1 = 2.0
    beta_2 = -1.5

    # Regressors
    x1 = np.random.randn(n_entities * n_periods) * 5 + 50
    x2 = np.random.randn(n_entities * n_periods) * 3 + 30

    # Common time-specific shocks (creates cross-sectional dependence)
    common_shocks = np.random.randn(n_periods) * 5
    common_shocks_expanded = np.tile(common_shocks, n_entities)

    # Entity fixed effects
    entity_effects = np.repeat(np.random.randn(n_entities) * 10, n_periods)

    # Idiosyncratic errors
    idiosyncratic = np.random.randn(n_entities * n_periods) * 2

    # Total error: common shock + idiosyncratic
    errors = common_shocks_expanded + idiosyncratic

    # Generate y
    y = beta_0 + beta_1 * x1 + beta_2 * x2 + entity_effects + errors

    entities = np.repeat(range(1, n_entities + 1), n_periods)
    times = np.tile(range(1, n_periods + 1), n_entities)

    data = pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    )

    return data


@pytest.fixture
def clean_panel_data():
    """
    Create panel data with no violations (clean data).

    - No serial correlation
    - Homoskedastic errors
    - No cross-sectional dependence
    """
    np.random.seed(125)

    n_entities = 50
    n_periods = 10

    # True parameters
    beta_0 = 10.0
    beta_1 = 2.0
    beta_2 = -1.5

    entities = np.repeat(range(1, n_entities + 1), n_periods)
    times = np.tile(range(1, n_periods + 1), n_entities)

    # Regressors
    x1 = np.random.randn(n_entities * n_periods) * 5 + 50
    x2 = np.random.randn(n_entities * n_periods) * 3 + 30

    # Entity fixed effects
    entity_effects = np.repeat(np.random.randn(n_entities) * 10, n_periods)

    # Clean i.i.d. errors
    errors = np.random.randn(n_entities * n_periods) * 5

    # Generate y
    y = beta_0 + beta_1 * x1 + beta_2 * x2 + entity_effects + errors

    data = pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    )

    return data


@pytest.fixture
def panel_for_mundlak():
    """
    Create panel data where RE assumption is violated.

    Regressors are correlated with entity effects (Mundlak should reject).
    """
    np.random.seed(127)

    n_entities = 50
    n_periods = 10

    # True parameters
    beta_0 = 10.0
    beta_1 = 2.0
    beta_2 = -1.5

    entities = np.repeat(range(1, n_entities + 1), n_periods)
    times = np.tile(range(1, n_periods + 1), n_entities)

    # Entity fixed effects
    entity_effects = np.random.randn(n_entities) * 10
    entity_effects_expanded = np.repeat(entity_effects, n_periods)

    # Regressors CORRELATED with entity effects
    x1_base = np.random.randn(n_entities * n_periods) * 5
    x1 = x1_base + 0.5 * entity_effects_expanded + 50

    x2_base = np.random.randn(n_entities * n_periods) * 3
    x2 = x2_base + 0.3 * entity_effects_expanded + 30

    # Clean errors
    errors = np.random.randn(n_entities * n_periods) * 5

    # Generate y
    y = beta_0 + beta_1 * x1 + beta_2 * x2 + entity_effects_expanded + errors

    data = pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    )

    return data
