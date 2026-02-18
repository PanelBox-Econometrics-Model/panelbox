"""
Data generators for the Visualization and Reports tutorial series.

All functions return a pandas DataFrame with a MultiIndex (entity, time) or
with columns ``entity`` and ``time`` that can be set as index by the caller.

Functions
---------
generate_panel_data
    Basic balanced panel with homoskedastic errors.
generate_heteroskedastic_panel
    Panel where error variance scales with a covariate (used in Notebook 02).
generate_autocorrelated_panel
    Panel with AR(1) errors within each entity (used in Notebook 02).
generate_spatial_panel
    Panel with cross-sectional dependence simulated via a common factor
    (used in Notebook 03).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_panel_data(
    n_individuals: int = 200,
    n_periods: int = 10,
    n_covariates: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a basic balanced panel dataset with homoskedastic errors.

    Parameters
    ----------
    n_individuals : int
        Number of cross-sectional units (entities). Default 200.
    n_periods : int
        Number of time periods. Default 10.
    n_covariates : int
        Number of regressors (x1, x2, ..., xk). Default 3.
    seed : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    pd.DataFrame
        Balanced panel with columns ``entity``, ``time``, ``x1``..``xk``, ``y``.
        MultiIndex (entity, time) is set as the DataFrame index.

    Examples
    --------
    >>> df = generate_panel_data(n_individuals=100, n_periods=5, seed=42)
    >>> df.shape
    (500, 3)
    >>> df.index.names
    FrozenList(['entity', 'time'])
    """
    rng = np.random.default_rng(seed)

    entities = np.repeat(np.arange(1, n_individuals + 1), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_individuals)

    # Entity fixed effects
    alpha = rng.normal(0, 1, n_individuals)
    alpha_full = np.repeat(alpha, n_periods)

    # Covariates
    X = rng.normal(0, 1, (n_individuals * n_periods, n_covariates))
    true_betas = np.array([1.5, -0.8, 0.5][:n_covariates])

    # Outcome
    epsilon = rng.normal(0, 1, n_individuals * n_periods)
    y = alpha_full + X @ true_betas + epsilon

    data = {"entity": entities, "time": times}
    for j in range(n_covariates):
        data[f"x{j + 1}"] = X[:, j]
    data["y"] = y

    df = pd.DataFrame(data).set_index(["entity", "time"])
    return df


def generate_heteroskedastic_panel(
    n_individuals: int = 200,
    n_periods: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a balanced panel with heteroskedastic errors.

    Error variance is proportional to ``|x1|``, creating a pattern detectable
    by Breusch-Pagan and White tests.  Used in Notebook 02 (visual diagnostics).

    Parameters
    ----------
    n_individuals : int
        Number of cross-sectional units. Default 200.
    n_periods : int
        Number of time periods. Default 10.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    pd.DataFrame
        Panel with columns ``x1``, ``x2``, ``y`` and MultiIndex (entity, time).
        Column ``sigma`` contains the true error standard deviation for reference.
    """
    rng = np.random.default_rng(seed)

    N = n_individuals * n_periods
    entities = np.repeat(np.arange(1, n_individuals + 1), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_individuals)

    alpha = np.repeat(rng.normal(0, 0.5, n_individuals), n_periods)

    x1 = rng.normal(2, 1, N)
    x2 = rng.normal(0, 1, N)

    # Heteroskedastic errors: sigma = 0.5 * |x1|
    sigma = 0.5 * np.abs(x1)
    epsilon = rng.normal(0, 1, N) * sigma

    y = alpha + 1.5 * x1 - 0.8 * x2 + epsilon

    df = pd.DataFrame(
        {"entity": entities, "time": times, "x1": x1, "x2": x2, "sigma": sigma, "y": y}
    ).set_index(["entity", "time"])
    return df


def generate_autocorrelated_panel(
    n_individuals: int = 200,
    n_periods: int = 10,
    rho: float = 0.7,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a balanced panel with AR(1) errors within each entity.

    Useful for demonstrating ACF/PACF plots and Durbin-Watson / Wooldridge
    tests for serial correlation.  Used in Notebook 02.

    Parameters
    ----------
    n_individuals : int
        Number of cross-sectional units. Default 200.
    n_periods : int
        Number of time periods. Default 10.
    rho : float
        AR(1) coefficient for within-entity serial correlation (|rho| < 1).
        Default 0.7.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    pd.DataFrame
        Panel with columns ``x1``, ``x2``, ``y``, ``epsilon`` and
        MultiIndex (entity, time).
    """
    rng = np.random.default_rng(seed)

    records = []
    for i in range(1, n_individuals + 1):
        alpha_i = rng.normal(0, 1)
        x1 = rng.normal(0, 1, n_periods)
        x2 = rng.normal(0, 1, n_periods)

        # AR(1) errors
        eps = np.empty(n_periods)
        eps[0] = rng.normal(0, 1) / np.sqrt(1 - rho**2)
        for t in range(1, n_periods):
            eps[t] = rho * eps[t - 1] + rng.normal(0, 1)

        y = alpha_i + 1.5 * x1 - 0.8 * x2 + eps

        for t in range(n_periods):
            records.append(
                {
                    "entity": i,
                    "time": t + 1,
                    "x1": x1[t],
                    "x2": x2[t],
                    "epsilon": eps[t],
                    "y": y[t],
                }
            )

    df = pd.DataFrame(records).set_index(["entity", "time"])
    return df


def generate_spatial_panel(
    n_individuals: int = 100,
    n_periods: int = 8,
    lambda_spatial: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a panel with cross-sectional dependence via a common factor.

    Simulates spatial-like dependence by adding a common time factor ``f_t``
    with entity-specific loadings ``lambda_i``.  Used in Notebook 03
    (advanced visualizations — cross-sectional correlation heatmap).

    Parameters
    ----------
    n_individuals : int
        Number of cross-sectional units. Default 100.
    n_periods : int
        Number of time periods. Default 8.
    lambda_spatial : float
        Strength of common factor loading variance. Default 0.4.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    pd.DataFrame
        Panel with columns ``x1``, ``y``, ``f_t`` (common factor) and
        MultiIndex (entity, time).
    """
    rng = np.random.default_rng(seed)

    # Common factor over time
    f_t = rng.normal(0, 1, n_periods)

    # Entity-specific factor loadings
    loadings = rng.normal(0, lambda_spatial, n_individuals)

    records = []
    for i in range(n_individuals):
        alpha_i = rng.normal(0, 0.5)
        x1 = rng.normal(0, 1, n_periods)
        idio_eps = rng.normal(0, 0.5, n_periods)

        y = alpha_i + 1.2 * x1 + loadings[i] * f_t + idio_eps

        for t in range(n_periods):
            records.append(
                {
                    "entity": i + 1,
                    "time": t + 1,
                    "x1": x1[t],
                    "f_t": f_t[t],
                    "loading": loadings[i],
                    "y": y[t],
                }
            )

    df = pd.DataFrame(records).set_index(["entity", "time"])
    return df


def make_var_data(n_obs: int = 200, n_vars: int = 3, seed: int = 42) -> pd.DataFrame:
    """Simulated VAR(1) data: GDP Growth, Inflation, Interest Rate (quarterly).

    Parameters
    ----------
    n_obs : int
        Number of observations (quarters). Default 200.
    n_vars : int
        Number of variables. Default 3.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex (quarterly) and columns
        ``gdp_growth``, ``inflation``, ``interest_rate``.
    """
    rng = np.random.default_rng(seed)
    A = np.array([[0.7, 0.1, -0.2], [0.1, 0.6, 0.1], [-0.1, 0.2, 0.5]])
    y = np.zeros((n_obs, n_vars))
    y[0] = rng.normal(0, 1, n_vars)
    for t in range(1, n_obs):
        y[t] = A @ y[t - 1] + rng.normal(0, 0.3, n_vars)
    dates = pd.date_range("1970Q1", periods=n_obs, freq="QE")
    return pd.DataFrame(y, index=dates, columns=["gdp_growth", "inflation", "interest_rate"])


def make_quantile_wage_panel(
    n_entities: int = 200, n_periods: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Panel wage data where the education effect is heterogeneous across quantiles.

    Effect is small at low quantiles and large at high quantiles.

    Parameters
    ----------
    n_entities : int
        Number of workers. Default 200.
    n_periods : int
        Number of time periods. Default 5.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    pd.DataFrame
        Panel with columns ``wage``, ``educ``, ``exper`` and
        MultiIndex (entity, time).
    """
    rng = np.random.default_rng(seed)
    records = []
    for i in range(1, n_entities + 1):
        educ = int(rng.integers(8, 20))
        exper = int(rng.integers(0, 30))
        ability = float(rng.normal(0, 1))
        for t in range(1, n_periods + 1):
            eps = float(rng.normal(0, 1))
            wage = 5 + 0.2 * educ + 0.05 * exper + ability + 0.1 * educ * ability + eps
            records.append(
                {
                    "entity": i,
                    "time": t,
                    "wage": max(wage, 0.0),
                    "educ": educ,
                    "exper": exper,
                }
            )
    df = pd.DataFrame(records).set_index(["entity", "time"])
    return df


def make_spatial_data(n_obs: int = 100, seed: int = 42) -> pd.DataFrame:
    """Spatial cross-section with coordinates and spatial autocorrelation.

    Simulates house prices with positive spatial autocorrelation.

    Parameters
    ----------
    n_obs : int
        Number of observations. Default 100.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``obs_id``, ``x``, ``y``, ``price``,
        ``size``, ``age``.
    """
    rng = np.random.default_rng(seed)
    x_coord = rng.uniform(0, 10, n_obs)
    y_coord = rng.uniform(0, 10, n_obs)
    price = rng.normal(200, 30, n_obs)
    for i in range(n_obs):
        for j in range(n_obs):
            dist = np.sqrt((x_coord[i] - x_coord[j]) ** 2 + (y_coord[i] - y_coord[j]) ** 2)
            if 0 < dist < 2:
                price[i] += 0.2 * (price[j] - 200)
    return pd.DataFrame(
        {
            "obs_id": np.arange(n_obs),
            "x": x_coord,
            "y": y_coord,
            "price": price,
            "size": rng.uniform(50, 200, n_obs),
            "age": rng.integers(1, 40, n_obs),
        }
    )


# Aliases for backward compatibility with planning docs
make_clean_panel = generate_panel_data
make_panel_with_heteroskedasticity = generate_heteroskedastic_panel
make_panel_with_autocorrelation = generate_autocorrelated_panel
