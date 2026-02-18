"""
Data generators for discrete choice tutorials.

This module provides functions to generate synthetic panel data for
discrete choice models, useful for demonstrations, testing, and
pedagogical purposes.

Functions:
    generate_labor_data              : Binary labor force participation data
    generate_multinomial_choice_data : Multinomial/conditional choice data
    generate_ordered_data            : Ordered categorical outcome data
    generate_dynamic_binary_data     : Dynamic binary choice with state dependence
    generate_transportation_choice_data : Transportation mode choice panel data (long format)
    generate_career_choice_data      : Career choice panel data (manual/technical/managerial)

Author: PanelBox Contributors
Date: 2026-02-16
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def generate_labor_data(
    n_individuals: int = 1000, n_periods: int = 5, seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic labor force participation panel data.

    Creates a balanced panel with binary labor force participation (lfp)
    as the dependent variable and standard demographic covariates.

    Parameters
    ----------
    n_individuals : int, default=1000
        Number of individuals in the panel.
    n_periods : int, default=5
        Number of time periods per individual.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel data with columns:
        - id       : Individual identifier
        - year     : Time period identifier
        - lfp      : Binary labor force participation (0/1)
        - age      : Age in years
        - educ     : Years of education
        - kids     : Number of children
        - married  : Marital status (0/1)
        - exper    : Years of work experience

    Examples
    --------
    >>> data = generate_labor_data(n_individuals=500, n_periods=4, seed=42)
    >>> data.groupby('id').size().unique()
    array([4])
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate individual-specific random effects
    alpha_i = np.random.normal(0, 1, n_individuals)

    # Generate time-varying covariates
    data = []
    for i in range(n_individuals):
        # Individual-specific starting values
        age_start = np.random.randint(25, 45)
        educ = np.random.randint(10, 18)
        exper_start = max(0, age_start - educ - 6 + np.random.randint(-2, 3))

        for t in range(n_periods):
            age = age_start + t
            exper = exper_start + t
            kids = np.random.poisson(0.3 * t)  # Kids accumulate over time
            married = 1 if np.random.random() < 0.5 + 0.02 * t else 0

            # True data generating process
            xb = (
                -3.0
                + 0.05 * age
                - 0.0005 * age**2
                + 0.15 * educ
                + 0.03 * exper
                - 0.5 * kids
                + 0.3 * married
                + alpha_i[i]
            )

            # Generate binary outcome via logit link
            prob = 1 / (1 + np.exp(-xb))
            lfp = 1 if np.random.random() < prob else 0

            data.append(
                {
                    "id": i + 1,
                    "year": t + 1,
                    "lfp": lfp,
                    "age": age,
                    "educ": educ,
                    "kids": kids,
                    "married": married,
                    "exper": exper,
                }
            )

    return pd.DataFrame(data)


def generate_multinomial_choice_data(
    n_obs: int = 5000,
    n_alternatives: int = 4,
    choice_specific_vars: bool = True,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic multinomial choice data.

    Creates data for multinomial logit or conditional logit models with
    multiple alternatives and both individual-specific and choice-specific
    covariates.

    Parameters
    ----------
    n_obs : int, default=5000
        Number of choice observations.
    n_alternatives : int, default=4
        Number of choice alternatives.
    choice_specific_vars : bool, default=True
        If True, generate choice-specific variables (conditional logit format).
        If False, only individual-specific variables (multinomial logit format).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple of pd.DataFrame
        If choice_specific_vars=True:
            (wide_data, long_data)
            - wide_data: One row per observation with choice and attributes
            - long_data: One row per observation-alternative combination
        If choice_specific_vars=False:
            (data, None)
            - data: One row per observation with individual variables

    Examples
    --------
    >>> wide, long = generate_multinomial_choice_data(n_obs=1000, seed=42)
    >>> wide.shape[0]
    1000
    >>> long.shape[0]
    4000
    """
    if seed is not None:
        np.random.seed(seed)

    if choice_specific_vars:
        # Generate choice-specific attributes (e.g., transportation modes)
        data_wide = []
        data_long = []

        for obs_id in range(n_obs):
            # Individual characteristics
            income = np.random.lognormal(10, 0.5)
            age = np.random.randint(18, 70)

            # Choice-specific attributes (e.g., cost and time for each mode)
            costs = np.random.uniform(1, 20, n_alternatives)
            times = np.random.uniform(10, 60, n_alternatives)

            # Utilities for each alternative
            utilities = []
            for alt in range(n_alternatives):
                # Alternative-specific constants
                asc = [0, 0.5, 1.0, -0.5][alt] if n_alternatives == 4 else 0

                u = (
                    asc
                    - 0.1 * costs[alt]
                    - 0.02 * times[alt]
                    + 0.0001 * income
                    + np.random.gumbel(0, 1)
                )
                utilities.append(u)

            # Choose alternative with highest utility
            choice = np.argmax(utilities)

            # Wide format (one row per observation)
            row_wide = {"id": obs_id + 1, "choice": choice, "income": income, "age": age}
            for alt in range(n_alternatives):
                row_wide[f"cost_{alt}"] = costs[alt]
                row_wide[f"time_{alt}"] = times[alt]
            data_wide.append(row_wide)

            # Long format (one row per observation-alternative)
            for alt in range(n_alternatives):
                data_long.append(
                    {
                        "id": obs_id + 1,
                        "alternative": alt,
                        "chosen": 1 if alt == choice else 0,
                        "cost": costs[alt],
                        "time": times[alt],
                        "income": income,
                        "age": age,
                    }
                )

        return pd.DataFrame(data_wide), pd.DataFrame(data_long)

    else:
        # Individual-specific variables only (standard multinomial logit)
        data = []
        for obs_id in range(n_obs):
            income = np.random.lognormal(10, 0.5)
            age = np.random.randint(18, 70)
            education = np.random.randint(8, 20)

            # Base utilities for each alternative
            u = np.zeros(n_alternatives)
            u[0] = 0  # Reference category
            u[1] = 0.5 + 0.0001 * income - 0.01 * age
            u[2] = 1.0 + 0.05 * education
            if n_alternatives > 3:
                u[3] = -0.5 - 0.02 * age + 0.0002 * income

            # Add Gumbel errors and choose
            u += np.random.gumbel(0, 1, n_alternatives)
            choice = np.argmax(u)

            data.append(
                {
                    "id": obs_id + 1,
                    "choice": choice,
                    "income": income,
                    "age": age,
                    "education": education,
                }
            )

        return pd.DataFrame(data), None


def generate_ordered_data(
    n_individuals: int = 800, n_periods: int = 4, n_categories: int = 4, seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic ordered categorical panel data.

    Creates a balanced panel with ordered categorical outcomes (e.g., credit
    ratings: poor, fair, good, excellent) using an ordered probit structure.

    Parameters
    ----------
    n_individuals : int, default=800
        Number of individuals in the panel.
    n_periods : int, default=4
        Number of time periods per individual.
    n_categories : int, default=4
        Number of ordered categories (minimum 3).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel data with columns:
        - id         : Individual identifier
        - year       : Time period identifier
        - rating     : Ordered categorical outcome (0 to n_categories-1)
        - income     : Income (continuous)
        - debt_ratio : Debt-to-income ratio
        - age        : Age in years

    Examples
    --------
    >>> data = generate_ordered_data(n_individuals=500, n_categories=5, seed=42)
    >>> data['rating'].nunique()
    5
    """
    if seed is not None:
        np.random.seed(seed)

    if n_categories < 3:
        raise ValueError("n_categories must be at least 3")

    # Generate cutpoints for ordered model
    cutpoints = np.linspace(-2, 2, n_categories - 1)

    # Generate individual-specific random effects
    alpha_i = np.random.normal(0, 0.5, n_individuals)

    data = []
    for i in range(n_individuals):
        # Individual-specific starting values
        income_start = np.random.lognormal(10.5, 0.5)
        age_start = np.random.randint(25, 60)

        for t in range(n_periods):
            age = age_start + t
            income = income_start * (1.03**t)  # Income grows over time
            debt_ratio = max(0.1, min(0.8, np.random.beta(2, 5) + 0.1 * t / n_periods))

            # Latent variable (credit worthiness)
            y_star = (
                0.0001 * income
                - 3.0 * debt_ratio
                + 0.01 * age
                + alpha_i[i]
                + np.random.normal(0, 1)
            )

            # Map latent variable to ordered categories
            rating = 0
            for j, cutpoint in enumerate(cutpoints):
                if y_star > cutpoint:
                    rating = j + 1

            data.append(
                {
                    "id": i + 1,
                    "year": t + 1,
                    "rating": rating,
                    "income": income,
                    "debt_ratio": debt_ratio,
                    "age": age,
                }
            )

    return pd.DataFrame(data)


def generate_dynamic_binary_data(
    n_individuals: int = 500,
    n_periods: int = 8,
    state_dependence: float = 1.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate dynamic binary choice panel data with state dependence.

    Creates a balanced panel where the lagged dependent variable affects
    the current choice, exhibiting state dependence (e.g., employment
    status, technology adoption).

    Parameters
    ----------
    n_individuals : int, default=500
        Number of individuals in the panel.
    n_periods : int, default=8
        Number of time periods per individual.
    state_dependence : float, default=1.5
        Coefficient on lagged dependent variable (higher = more persistence).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel data with columns:
        - id       : Individual identifier
        - year     : Time period identifier
        - y        : Binary outcome (0/1)
        - y_lag    : Lagged outcome
        - x1, x2   : Continuous covariates

    Examples
    --------
    >>> data = generate_dynamic_binary_data(n_individuals=300, seed=42)
    >>> data[data['year'] > 1]['y_lag'].isna().sum()
    0
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate individual-specific random effects and initial conditions
    alpha_i = np.random.normal(0, 1, n_individuals)

    data = []
    for i in range(n_individuals):
        # Individual-specific covariates (time-invariant)
        x1_i = np.random.normal(0, 1)

        y_lag = None  # No lagged value in first period

        for t in range(n_periods):
            # Time-varying covariate
            x2_t = np.random.normal(0, 1)

            # Latent utility
            if t == 0:
                # Initial condition (no lagged y)
                xb = -1.0 + 0.5 * x1_i + 0.3 * x2_t + alpha_i[i]
            else:
                # Include state dependence
                xb = -1.0 + state_dependence * y_lag + 0.5 * x1_i + 0.3 * x2_t + alpha_i[i]

            # Generate binary outcome
            prob = 1 / (1 + np.exp(-xb))
            y = 1 if np.random.random() < prob else 0

            data.append(
                {
                    "id": i + 1,
                    "year": t + 1,
                    "y": y,
                    "y_lag": y_lag if y_lag is not None else np.nan,
                    "x1": x1_i,
                    "x2": x2_t,
                }
            )

            y_lag = y  # Update lagged value

    return pd.DataFrame(data)


def generate_transportation_choice_data(
    n_individuals: int = 800, n_periods: int = 5, seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic transportation mode choice panel data in long format.

    Creates a balanced panel where each individual chooses among four transport
    modes (car, bus, metro, bike) in each period. Attributes like cost and time
    vary by individual and alternative, while income and distance are
    individual-specific.

    Parameters
    ----------
    n_individuals : int, default=800
        Number of individuals in the panel.
    n_periods : int, default=5
        Number of time periods per individual.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-format panel data with columns:
        - id          : Individual identifier (int)
        - year        : Time period (int)
        - mode        : Transport mode ('car', 'bus', 'metro', 'bike')
        - choice      : 1 if chosen, 0 otherwise (int)
        - cost        : Cost in R$ (float)
        - time        : Travel time in minutes (float)
        - reliability : Reliability score 1-5 (float)
        - comfort     : Comfort score 1-5 (float)
        - income      : Individual income (float, individual-specific)
        - distance    : Commute distance in km (float, individual-specific)

    Notes
    -----
    For each (id, year), exactly one mode has choice=1.
    The data generating process uses a conditional logit model with
    alternative-specific constants and known coefficient values.

    Examples
    --------
    >>> data = generate_transportation_choice_data(n_individuals=100, n_periods=3, seed=42)
    >>> data.groupby(['id', 'year'])['choice'].sum().eq(1).all()
    True
    """
    if seed is not None:
        np.random.seed(seed)

    modes = ["car", "bus", "metro", "bike"]
    n_modes = len(modes)

    # Alternative-specific constants (relative to bike=reference)
    asc = {"car": 0.2, "bus": 0.8, "metro": -0.1, "bike": 0.0}

    # True parameters for the DGP
    gamma_cost = -0.04  # Higher cost -> lower utility
    gamma_time = -0.025  # More time -> lower utility
    gamma_reliability = 0.18  # Higher reliability -> higher utility
    gamma_comfort = 0.12  # Higher comfort -> higher utility
    # Income interactions (relative to bike)
    delta_income = {"car": 0.00002, "bus": -0.00001, "metro": 0.00001, "bike": 0.0}

    rows = []

    for i in range(1, n_individuals + 1):
        # Individual-specific characteristics (time-invariant)
        income = np.random.lognormal(mean=8.5, sigma=0.6) * 10  # ~R$5000-R$15000
        distance = np.random.uniform(3, 40)  # 3-40 km

        for t in range(1, n_periods + 1):
            # Generate alternative-specific attributes
            # Cost depends on mode and distance
            costs = {
                "car": max(5, distance * np.random.uniform(0.8, 1.5) + np.random.normal(5, 2)),
                "bus": max(3, 4.40 + np.random.normal(0, 0.5)),
                "metro": max(3, 4.80 + distance * 0.05 + np.random.normal(0, 0.5)),
                "bike": max(0.5, distance * np.random.uniform(0.02, 0.08)),
            }

            # Time depends on mode and distance
            times = {
                "car": max(5, distance * np.random.uniform(1.5, 3.0) + np.random.normal(5, 3)),
                "bus": max(10, distance * np.random.uniform(2.5, 4.5) + np.random.normal(10, 5)),
                "metro": max(8, distance * np.random.uniform(1.2, 2.5) + np.random.normal(8, 3)),
                "bike": max(10, distance * np.random.uniform(3.0, 5.0) + np.random.normal(0, 3)),
            }

            # Reliability (1-5 scale)
            reliabilities = {
                "car": np.clip(np.random.normal(3.5, 0.7), 1, 5),
                "bus": np.clip(np.random.normal(2.8, 0.8), 1, 5),
                "metro": np.clip(np.random.normal(4.0, 0.6), 1, 5),
                "bike": np.clip(np.random.normal(3.2, 0.9), 1, 5),
            }

            # Comfort (1-5 scale)
            comforts = {
                "car": np.clip(np.random.normal(4.2, 0.5), 1, 5),
                "bus": np.clip(np.random.normal(2.5, 0.7), 1, 5),
                "metro": np.clip(np.random.normal(3.3, 0.6), 1, 5),
                "bike": np.clip(np.random.normal(2.0, 0.8), 1, 5),
            }

            # Compute utilities
            utilities = {}
            for mode in modes:
                u = (
                    asc[mode]
                    + gamma_cost * costs[mode]
                    + gamma_time * times[mode]
                    + gamma_reliability * reliabilities[mode]
                    + gamma_comfort * comforts[mode]
                    + delta_income[mode] * income
                    + np.random.gumbel(0, 1)
                )  # Type I extreme value error
                utilities[mode] = u

            # Choose mode with highest utility
            chosen_mode = max(utilities, key=utilities.get)

            # Build rows
            for mode in modes:
                rows.append(
                    {
                        "id": i,
                        "year": t,
                        "mode": mode,
                        "choice": 1 if mode == chosen_mode else 0,
                        "cost": round(costs[mode], 2),
                        "time": round(times[mode], 1),
                        "reliability": round(reliabilities[mode], 1),
                        "comfort": round(comforts[mode], 1),
                        "income": round(income, 2),
                        "distance": round(distance, 1),
                    }
                )

    df = pd.DataFrame(rows)
    return df


def generate_career_choice_data(
    n_individuals: int = 1000,
    n_periods: int = 5,
    n_alternatives: int = 3,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic career choice panel data.

    Creates a balanced panel where individuals choose among career paths
    (manual, technical, managerial) based on individual characteristics.
    Designed for Multinomial Logit tutorials.

    Parameters
    ----------
    n_individuals : int, default=1000
        Number of individuals in the panel.
    n_periods : int, default=5
        Number of time periods per individual.
    n_alternatives : int, default=3
        Number of career alternatives (fixed at 3).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel data with columns:
        - id       : Individual identifier (int)
        - year     : Time period (int)
        - career   : Career choice: 0=manual, 1=technical, 2=managerial (int)
        - educ     : Years of education (float)
        - exper    : Years of experience (float)
        - age      : Age in years (float)
        - female   : 1=female, 0=male (int)
        - income   : Current income in R$ (float)
        - urban    : 1=urban, 0=rural (int)

    Notes
    -----
    The data generating process uses a multinomial logit structure where:
    - Career 0 (manual) is the base/reference category
    - Education strongly favors technical and managerial careers
    - Experience favors managerial careers
    - Gender affects career selection (to illustrate group differences)
    - Urban location favors technical and managerial careers

    Examples
    --------
    >>> data = generate_career_choice_data(n_individuals=500, n_periods=4, seed=42)
    >>> data['career'].value_counts().sort_index()
    0    ...
    1    ...
    2    ...
    """
    if seed is not None:
        np.random.seed(seed)

    # True parameters for DGP (relative to career=0 manual)
    # Career 1 (technical) vs manual
    beta_1 = {
        "const": -0.8,
        "educ": 0.10,
        "exper": 0.02,
        "age": -0.01,
        "female": -0.25,
        "income_effect": 0.00002,
        "urban": 0.30,
    }
    # Career 2 (managerial) vs manual
    beta_2 = {
        "const": -3.5,
        "educ": 0.25,
        "exper": 0.05,
        "age": 0.005,
        "female": -0.40,
        "income_effect": 0.00004,
        "urban": 0.40,
    }

    # Individual-level random effects (unobserved heterogeneity)
    alpha_i = np.random.normal(0, 0.5, n_individuals)

    rows = []

    for i in range(1, n_individuals + 1):
        # Time-invariant characteristics
        female = int(np.random.random() < 0.45)
        urban = int(np.random.random() < 0.65)
        educ_base = np.random.normal(12, 3)
        educ_base = np.clip(educ_base, 6, 22)
        age_start = np.random.randint(22, 50)
        exper_start = max(0, age_start - educ_base - 6 + np.random.randint(-2, 4))

        for t in range(1, n_periods + 1):
            age = age_start + t - 1
            exper = exper_start + t - 1
            educ = educ_base + (0.5 if t > 3 and np.random.random() < 0.1 else 0)

            # Income depends on career history (simplified)
            base_income = 2000 + 300 * educ + 150 * exper + 500 * urban
            income = base_income * np.exp(np.random.normal(0, 0.2))

            # Compute utilities
            u0 = 0.0  # Base category (manual)

            u1 = (
                beta_1["const"]
                + beta_1["educ"] * educ
                + beta_1["exper"] * exper
                + beta_1["age"] * age
                + beta_1["female"] * female
                + beta_1["urban"] * urban
                + alpha_i[i - 1] * 0.5
            )

            u2 = (
                beta_2["const"]
                + beta_2["educ"] * educ
                + beta_2["exper"] * exper
                + beta_2["age"] * age
                + beta_2["female"] * female
                + beta_2["urban"] * urban
                + alpha_i[i - 1] * 0.8
            )

            # Add Gumbel errors (Type I extreme value)
            utilities = np.array([u0, u1, u2])
            utilities += np.random.gumbel(0, 1, 3)

            # Choose career with highest utility
            career = int(np.argmax(utilities))

            rows.append(
                {
                    "id": i,
                    "year": t,
                    "career": career,
                    "educ": round(educ, 1),
                    "exper": round(exper, 1),
                    "age": round(float(age), 1),
                    "female": female,
                    "income": round(income, 2),
                    "urban": urban,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Example usage
    print("Generating example datasets...")

    labor = generate_labor_data(n_individuals=100, n_periods=3, seed=42)
    print(f"\nLabor data: {labor.shape}")
    print(labor.head(10))

    multi_wide, multi_long = generate_multinomial_choice_data(n_obs=100, seed=42)
    print(f"\nMultinomial data (wide): {multi_wide.shape}")
    print(f"Multinomial data (long): {multi_long.shape}")

    ordered = generate_ordered_data(n_individuals=100, n_periods=3, seed=42)
    print(f"\nOrdered data: {ordered.shape}")
    print(ordered["rating"].value_counts().sort_index())

    dynamic = generate_dynamic_binary_data(n_individuals=100, n_periods=5, seed=42)
    print(f"\nDynamic binary data: {dynamic.shape}")
    print(f"Proportion y=1: {dynamic['y'].mean():.3f}")

    transport = generate_transportation_choice_data(n_individuals=100, n_periods=3, seed=42)
    print(f"\nTransportation data: {transport.shape}")
    print(f"Modes: {transport['mode'].unique()}")
    print(f"Choice shares:")
    print(transport[transport["choice"] == 1]["mode"].value_counts(normalize=True))
