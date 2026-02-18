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
    generate_credit_rating_data      : Credit rating panel data (ordered categories)
    generate_labor_dynamics_data     : Dynamic labor participation with state dependence (probit)
    generate_work_mode_data          : Work mode choice panel (on-site/hybrid/remote)

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


def generate_credit_rating_data(
    n_firms: int = 600,
    n_periods: int = 5,
    n_categories: int = 4,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate synthetic credit rating panel data.

    Creates a balanced panel of firms with ordered credit ratings
    (poor, fair, good, excellent) using an ordered logit latent
    variable structure with firm-level random effects.

    Parameters
    ----------
    n_firms : int, default=600
        Number of firms in the panel.
    n_periods : int, default=5
        Number of time periods per firm.
    n_categories : int, default=4
        Number of ordered rating categories (minimum 3).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel data with columns:
        - id            : Firm identifier (int)
        - year          : Time period (int)
        - rating        : Credit rating 0=poor, 1=fair, 2=good, 3=excellent (int)
        - income        : Log firm income (float)
        - debt_ratio    : Debt-to-assets ratio (float)
        - age           : Firm age in years (float)
        - size          : Log total assets (float)
        - profitability : Return on assets (float)
        - sector        : Industry sector 1-5 (int)

    Examples
    --------
    >>> df = generate_credit_rating_data(n_firms=600, n_periods=5, seed=42)
    >>> df['rating'].value_counts().sort_index()
    0    ...
    1    ...
    2    ...
    3    ...
    """
    if seed is not None:
        np.random.seed(seed)

    if n_categories < 3:
        raise ValueError("n_categories must be at least 3")

    # True cutpoints for latent variable model (centered to balance categories)
    cutpoints = np.array([-0.8, 0.4, 1.6])[: n_categories - 1]

    # Firm-level random effects (persistent quality / reputation)
    alpha_i = np.random.normal(0, 0.6, n_firms)

    # Sector assignment (time-invariant)
    sectors = np.random.choice([1, 2, 3, 4, 5], size=n_firms)

    rows = []

    for i in range(n_firms):
        # Firm-specific baseline characteristics
        income_base = np.random.normal(10.5, 1.0)
        size_base = np.random.normal(8.0, 1.5)
        age_start = np.random.uniform(3, 50)
        debt_base = np.random.beta(3, 5)  # Right-skewed, mean ~0.375

        for t in range(n_periods):
            # Time-varying covariates with persistence
            income = income_base + 0.02 * t + np.random.normal(0, 0.15)
            size = size_base + 0.01 * t + np.random.normal(0, 0.1)
            age = age_start + t
            debt_ratio = np.clip(debt_base + 0.01 * t + np.random.normal(0, 0.05), 0.05, 0.95)
            profitability = np.clip(
                0.08 + 0.01 * (income - 10) - 0.05 * debt_ratio + np.random.normal(0, 0.03),
                -0.10,
                0.30,
            )

            # Latent credit quality (ordered logit DGP)
            # Coefficients chosen so Xβ is centered near middle cutpoints
            y_star = (
                -2.8  # intercept to center latent variable
                + 0.30 * income  # β_income
                - 1.8 * debt_ratio  # β_debt (negative: more debt → worse)
                + 0.01 * age  # β_age
                + 0.10 * size  # β_size
                + 2.0 * profitability  # β_profitability
                + alpha_i[i]
                + np.random.logistic(0, 1)  # Logistic error for ordered logit
            )

            # Map latent variable to ordered categories via cutpoints
            rating = 0
            for j, kappa in enumerate(cutpoints):
                if y_star > kappa:
                    rating = j + 1

            rows.append(
                {
                    "id": i + 1,
                    "year": 2015 + t,
                    "rating": rating,
                    "income": round(income, 4),
                    "debt_ratio": round(debt_ratio, 4),
                    "age": round(age, 1),
                    "size": round(size, 4),
                    "profitability": round(profitability, 4),
                    "sector": int(sectors[i]),
                }
            )

    return pd.DataFrame(rows)


def generate_labor_dynamics_data(
    n_individuals: int = 1500,
    n_periods: int = 10,
    true_gamma: float = 0.4,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate dynamic labor force participation panel data with state dependence.

    Creates a balanced panel of women tracked over multiple years where
    employment status depends on lagged employment (state dependence) and
    persistent unobserved heterogeneity. Designed for dynamic discrete
    choice tutorials (Wooldridge 2005 approach).

    The data generating process uses a probit structure:
        y*_it = X_it'β + γ * y_{i,t-1} + α_i + ε_it,  ε ~ N(0,1)
        y_it = 1[y*_it > 0]

    Parameters
    ----------
    n_individuals : int, default=1500
        Number of women in the panel.
    n_periods : int, default=10
        Number of time periods per individual.
    true_gamma : float, default=0.4
        True state dependence parameter (coefficient on lagged employment).
        Higher values mean stronger persistence from past employment.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel data with columns:
        - id       : Individual identifier (int)
        - year     : Time period (int)
        - employed : Employment status 1=employed, 0=not (int)
        - age      : Age in years (float)
        - educ     : Years of education (float)
        - kids     : Number of young children (int)
        - married  : Marital status 1=married, 0=not (int)
        - exper    : Years of labor experience (float)
        - husbinc  : Husband's income in thousands (float)

    Notes
    -----
    The DGP ensures:
    - Substantial state dependence (γ ≈ 0.4 by default)
    - Meaningful unobserved heterogeneity (σ_α ≈ 0.8)
    - Both sources contribute to observed persistence
    - Initial conditions are correlated with α_i (Heckman problem)

    Examples
    --------
    >>> data = generate_labor_dynamics_data(n_individuals=500, n_periods=8, seed=42)
    >>> data.shape
    (4000, 9)
    >>> data.groupby('id').size().unique()
    array([8])
    """
    if seed is not None:
        np.random.seed(seed)

    # Unobserved heterogeneity (persistent individual effect)
    sigma_alpha = 0.8
    alpha_i = np.random.normal(0, sigma_alpha, n_individuals)

    # True coefficients (probit scale)
    beta_age = 0.02
    beta_age_sq = -0.0003
    beta_educ = 0.06
    beta_kids = -0.25
    beta_married = 0.10
    beta_exper = 0.015
    beta_husbinc = -0.015
    intercept = -1.5

    rows = []

    for i in range(n_individuals):
        # Time-invariant characteristics
        educ = np.clip(np.random.normal(12.5, 2.5), 8, 20)
        age_start = np.random.randint(22, 42)
        exper_start = max(0, age_start - educ - 6 + np.random.randint(-2, 4))
        married_base = np.random.random() < 0.65
        husbinc_base = np.random.lognormal(2.8, 0.6) if married_base else 0.0

        y_prev = None  # No lag for first period

        for t in range(n_periods):
            year = 2000 + t
            age = age_start + t
            exper = exper_start + t * 0.8  # Experience grows slower than time
            kids = max(0, np.random.poisson(max(0, 1.5 - 0.03 * age + 0.1 * t)))
            married = int(married_base if np.random.random() < 0.9 else not married_base)
            husbinc = (
                max(0, husbinc_base * (1 + 0.02 * t) + np.random.normal(0, 1.5)) if married else 0.0
            )

            # Latent utility (probit)
            xb = (
                intercept
                + beta_age * age
                + beta_age_sq * age**2
                + beta_educ * educ
                + beta_kids * kids
                + beta_married * married
                + beta_exper * exper
                + beta_husbinc * husbinc
                + alpha_i[i]
            )

            if t == 0:
                # Initial condition: correlated with alpha_i
                # Use a reduced-form for the initial period (no lag)
                xb_init = xb + 0.3 * alpha_i[i]  # Extra correlation
            else:
                # Dynamic model: include state dependence
                xb = xb + true_gamma * y_prev
                xb_init = xb

            # Probit: Phi(xb) is the probability
            epsilon = np.random.normal(0, 1)
            employed = int(xb_init + epsilon > 0)

            rows.append(
                {
                    "id": i + 1,
                    "year": year,
                    "employed": employed,
                    "age": round(float(age), 1),
                    "educ": round(educ, 1),
                    "kids": int(kids),
                    "married": married,
                    "exper": round(exper, 1),
                    "husbinc": round(husbinc, 2),
                }
            )

            y_prev = employed

    return pd.DataFrame(rows)


def generate_work_mode_data(
    n_workers: int = 2000,
    n_years: int = 5,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate synthetic work mode choice panel data (post-pandemic context).

    Creates a balanced panel of workers choosing among three work modes
    (on-site, hybrid, remote) over 2019-2023. The DGP incorporates a
    pandemic shift (2020-2021), state dependence (past remote experience
    affects future choices), and unobserved heterogeneity in remote
    work preferences.

    Parameters
    ----------
    n_workers : int, default=2000
        Number of workers in the panel.
    n_years : int, default=5
        Number of years (2019 through 2019+n_years-1).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel data with columns:
        - worker_id  : Worker identifier (int)
        - year       : Year 2019-2023 (int)
        - mode       : Work mode: 0=on-site, 1=hybrid, 2=remote (int)
        - prod_remote: Remote productivity score 1-10 (float)
        - commute    : Commute time in minutes (float)
        - kids       : Number of young children (int)
        - age        : Age in years (float)
        - educ       : Years of education (float)
        - income     : Monthly income (float)
        - sector     : Industry sector 1-5 (int)
        - firm_size  : Firm size: 1=small, 2=medium, 3=large (int)
        - tech_job   : 1=technology sector, 0=other (int)

    Notes
    -----
    The DGP uses a multinomial logit structure with:
    - Mode 0 (on-site) as the base category
    - Pandemic effect (2020-2021) shifting utilities toward remote/hybrid
    - State dependence: being remote in t-1 increases P(remote) in t
    - Unobserved heterogeneity: individual-specific remote preference

    Examples
    --------
    >>> data = generate_work_mode_data(n_workers=500, n_years=5, seed=42)
    >>> data.shape
    (2500, 12)
    >>> data['mode'].value_counts().sort_index()
    0    ...
    1    ...
    2    ...
    """
    if seed is not None:
        np.random.seed(seed)

    # Unobserved heterogeneity: individual preference for remote work
    alpha_remote = np.random.normal(0, 0.7, n_workers)
    alpha_hybrid = np.random.normal(0, 0.5, n_workers)

    # True parameters for DGP (relative to mode=0, on-site)
    # Utility for hybrid (mode=1) vs on-site
    beta_hybrid = {
        "const": -1.0,
        "prod_remote": 0.10,
        "commute": 0.008,
        "kids": 0.15,
        "age": -0.01,
        "educ": 0.03,
        "tech_job": 0.30,
    }
    # Utility for remote (mode=2) vs on-site
    beta_remote = {
        "const": -2.5,
        "prod_remote": 0.25,
        "commute": 0.015,
        "kids": -0.05,
        "age": -0.02,
        "educ": 0.05,
        "tech_job": 0.50,
    }

    # State dependence (experience with remote/hybrid)
    gamma_remote = 0.80  # Being remote in t-1 -> higher P(remote in t)
    gamma_hybrid = 0.50  # Being hybrid in t-1 -> higher P(hybrid in t)

    # Pandemic effect (shifts for 2020-2021)
    pandemic_shift_remote = {2019: 0.0, 2020: 1.8, 2021: 1.2, 2022: 0.6, 2023: 0.3}
    pandemic_shift_hybrid = {2019: 0.0, 2020: 1.0, 2021: 0.8, 2022: 0.5, 2023: 0.3}

    rows = []

    for i in range(n_workers):
        # Time-invariant characteristics
        age_start = np.random.randint(25, 55)
        educ = np.clip(np.random.normal(14, 3), 8, 22)
        sector = np.random.choice([1, 2, 3, 4, 5])
        tech_job = 1 if sector == 1 or (sector == 2 and np.random.random() < 0.3) else 0
        firm_size = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
        base_commute = np.random.lognormal(3.2, 0.5)  # ~25 min median
        base_income = np.exp(
            7.5 + 0.08 * educ + 0.02 * age_start + 0.3 * tech_job + np.random.normal(0, 0.3)
        )

        # Remote productivity (partially driven by tech_job, educ, individual ability)
        prod_remote_base = np.clip(
            4.0 + 1.5 * tech_job + 0.15 * educ + alpha_remote[i] * 0.5 + np.random.normal(0, 1.0),
            1,
            10,
        )

        y_prev = 0  # Start on-site in pre-pandemic
        kids_val = max(0, np.random.poisson(0.8))

        for t in range(n_years):
            year = 2019 + t
            age = age_start + t
            income = base_income * (1.03**t) * np.exp(np.random.normal(0, 0.05))
            commute = max(5, base_commute + np.random.normal(0, 5))

            # Kids can change slightly over time
            if t > 0 and np.random.random() < 0.08:
                kids_val = max(0, kids_val + np.random.choice([-1, 1], p=[0.3, 0.7]))

            # Productivity evolves (learning effect if was remote before)
            prod_remote = np.clip(
                prod_remote_base + 0.3 * (y_prev == 2) + np.random.normal(0, 0.3), 1, 10
            )

            # Pandemic shift
            p_shift_r = pandemic_shift_remote.get(year, 0.0)
            p_shift_h = pandemic_shift_hybrid.get(year, 0.0)

            # State dependence
            sd_remote = gamma_remote * (1 if y_prev == 2 else 0)
            sd_hybrid = gamma_hybrid * (1 if y_prev == 1 else 0)
            # Cross-effects: hybrid experience also slightly increases P(remote)
            sd_remote += 0.2 * (1 if y_prev == 1 else 0)

            # Firm size effect: large firms adopted hybrid/remote faster
            firm_effect_h = 0.2 * (firm_size - 1)
            firm_effect_r = 0.15 * (firm_size - 1)

            # Compute utilities
            u_onsite = 0.0  # base

            u_hybrid = (
                beta_hybrid["const"]
                + beta_hybrid["prod_remote"] * prod_remote
                + beta_hybrid["commute"] * commute
                + beta_hybrid["kids"] * kids_val
                + beta_hybrid["age"] * age
                + beta_hybrid["educ"] * educ
                + beta_hybrid["tech_job"] * tech_job
                + p_shift_h
                + sd_hybrid
                + firm_effect_h
                + alpha_hybrid[i]
            )

            u_remote = (
                beta_remote["const"]
                + beta_remote["prod_remote"] * prod_remote
                + beta_remote["commute"] * commute
                + beta_remote["kids"] * kids_val
                + beta_remote["age"] * age
                + beta_remote["educ"] * educ
                + beta_remote["tech_job"] * tech_job
                + p_shift_r
                + sd_remote
                + firm_effect_r
                + alpha_remote[i]
            )

            # Add Gumbel errors (Type I extreme value)
            utilities = np.array([u_onsite, u_hybrid, u_remote])
            utilities += np.random.gumbel(0, 1, 3)

            mode = int(np.argmax(utilities))

            rows.append(
                {
                    "worker_id": i + 1,
                    "year": year,
                    "mode": mode,
                    "prod_remote": round(prod_remote, 2),
                    "commute": round(commute, 1),
                    "kids": int(kids_val),
                    "age": round(float(age), 1),
                    "educ": round(educ, 1),
                    "income": round(income, 2),
                    "sector": int(sector),
                    "firm_size": int(firm_size),
                    "tech_job": int(tech_job),
                }
            )

            y_prev = mode

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
